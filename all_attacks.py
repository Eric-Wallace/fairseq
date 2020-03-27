import numpy as np
from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer
import nltk
from nltk.corpus import wordnet
import attack_utils
import all_attack_utils

# find the position of the start and end of the original_output_token and replaces it with desired_output_token
# desired_output_token can be shorter, longer, or the same length as original_output_token
def find_and_replace_target(samples, original_output_token, desired_output_token):
    target_mask = []
    start_pos = None
    end_pos = None
    for idx, current_token in enumerate(samples['target'].cpu()[0]):
        if current_token == original_output_token[0]: # TODO, the logic here will fail when a BPE id is repeated
            start_pos = idx
        if current_token == original_output_token[-1]:
            end_pos = idx
    if start_pos is None or end_pos is None:
        exit('find and replace target failed to find token')

    last_tokens_of_target = deepcopy(samples['target'][0][end_pos+1:])
    new_start = torch.cat((samples['target'][0][0:start_pos], desired_output_token.cuda()), dim=0)
    new_target = torch.cat((new_start, last_tokens_of_target), dim=0)
    target_mask = [0] * start_pos + [1] * len(desired_output_token) + [0] * (len(new_target) - len(desired_output_token) - start_pos)
    samples['target'] = new_target.unsqueeze(0)
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:], samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)
    return samples, target_mask

def main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    # make sure everything is reset before loading the model
    args.reset_optimizer = True
    args.reset_meters = True
    args.reset_dataloader = True
    args.reset_lr_scheduler = True
    args.path = args.restore_file
    utils.import_user_module(args)
    args.max_sentences_valid = 1  # batch size 1 at the moment

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # setup task, model, loss function, and trainer
    task = tasks.setup_task(args)
    if not args.interactive_attacks:
        for valid_sub_split in args.valid_subset.split(','): # load validation data
            task.load_dataset(valid_sub_split, combine=False, epoch=0)
    models, _= checkpoint_utils.load_model_ensemble(args.path.split(':'), arg_overrides={}, task=task)
    model = models[0]

    if torch.cuda.is_available() and not args.cpu:
        assert torch.cuda.device_count() == 1 # only works on 1 GPU for now
        torch.cuda.set_device(0)
        model.cuda()
    args.beam = 1 # beam size 1 for now
    model.make_generation_fast_(beamable_mm_beam_size=args.beam, need_attn=False)

    criterion = task.build_criterion(args)
    trainer = Trainer(args, task, model, criterion)
    generator = task.build_generator(args)
    # print(args); print(task); print(model); print(criterion); print(generator)

    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    all_attack_utils.add_hooks(trainer.get_model(), bpe_vocab_size) # add gradient hooks to embeddings
    embedding_weight = all_attack_utils.get_embedding_weight(trainer.get_model(), bpe_vocab_size) # save the embedding matrix
    if not args.interactive_attacks:
        subset = args.valid_subset.split(',')[0] # only one validation subset handled
        itr = trainer.task.get_batch_iterator(dataset=trainer.task.dataset(subset),
                                      max_tokens=args.max_tokens_valid,
                                      max_sentences=args.max_sentences_valid,
                                      max_positions=utils.resolve_max_positions(
                                      trainer.task.max_positions(),
                                      trainer.get_model().max_positions(),),
                                      ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                                      required_batch_size_multiple=args.required_batch_size_multiple,
                                      seed=args.seed,
                                      num_shards=args.distributed_world_size,
                                      shard_id=args.distributed_rank,
                                      num_workers=args.num_workers,).next_epoch_itr(shuffle=False)
    else:
        itr = [None] * 100000  # a fake dataset to go through, overwritten when doing interactive attacks

    # Handle BPE
    bpe = encoders.build_bpe(args)
    assert bpe is not None
    attack(args, trainer, generator, embedding_weight, itr, bpe)

def attack(args, trainer, generator, embedding_weight, itr, bpe):
    num_samples_changed = 0.0
    num_total_samples = 0.0
    num_tokens_changed = 0.0
    total_num_tokens = 0.0
    for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
        changed_positions = None
        if args.targeted_flips:
            changed_positions = targeted_flips(samples, args, trainer, generator, embedding_weight, itr, bpe, i)        
        elif args.malicious_appends:
            malicious_appends(samples, args, trainer, generator, embedding_weight, itr, bpe, i)
        else:
            exit("pick an attack mode using --targeted-flips, --malicious-appends, or --universal-triggers")
        if changed_positions is None:
            continue

        num_total_samples += 1.0        
        if any(changed_positions):
            num_samples_changed += 1.0
            num_tokens_changed += sum(changed_positions)
            total_num_tokens += len(changed_positions)        
    if num_total_samples > 0.0:
         print('\n\n\n')
         print('Total Num Samples', num_total_samples)
         print('Percent Samples Changed', num_samples_changed / num_total_samples)
         print('Percent Tokens Changed', num_tokens_changed / total_num_tokens)


def targeted_flips(samples, args, trainer, generator, embedding_weight, itr, bpe, i):
    assert args.interactive_attacks # only interactive for now
    if args.interactive_attacks: # get user input and build samples
        samples = attack_utils.get_user_input(trainer, bpe)    
    if samples is None:
        return None

    samples, original_prediction = all_attack_utils.run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=False)
    if args.interactive_attacks:
        print('Current Translation ', bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(original_prediction), None)))
        original_output_token = input('Enter the target token ')
        desired_output_token = input('Enter desired target token ')
        adversarial_token_blacklist_string = input('Enter optional space seperated blacklist of invalid adversarial words ')
        untouchable_token_blacklist_string = input('Enter optional space seperated blacklist of source words to keep ')

        # -1 strips off <eos> token
        original_output_token = trainer.task.target_dictionary.encode_line(bpe.encode(original_output_token)).long()[0:-1]
        desired_output_token = trainer.task.target_dictionary.encode_line(bpe.encode(desired_output_token)).long()[0:-1]
        print("Original Output Len", len(original_output_token), "Desired Output Len", len(desired_output_token))        

        # don't change any of these tokens in the input
        untouchable_token_blacklist = []
        if untouchable_token_blacklist_string is not None and untouchable_token_blacklist_string != '' and untouchable_token_blacklist_string != '\n':
            untouchable_token_blacklist_string = untouchable_token_blacklist_string.split(' ')
            for token in untouchable_token_blacklist_string:
                token = trainer.task.source_dictionary.encode_line(bpe.encode(token)).long()[0:-1]                
                untouchable_token_blacklist.extend(token)

        # don't insert any of these tokens (or their synonyms) into the source
        adversarial_token_blacklist = []
        adversarial_token_blacklist.extend(desired_output_token) # don't let the attack put these words in
        if adversarial_token_blacklist_string is not None and adversarial_token_blacklist_string != '' and adversarial_token_blacklist_string != '\n':
            adversarial_token_blacklist_string = adversarial_token_blacklist_string.split(' ')
            synonyms = set()
            for token in adversarial_token_blacklist_string:
                token = trainer.task.source_dictionary.encode_line(bpe.encode(token)).long()[0:-1]
                if len(token) == 1:
                    adversarial_token_blacklist.append(token)
                    for syn in wordnet.synsets(bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(token), None))): # don't add any synonyms either
                        for l in syn.lemmas():
                            synonyms.add(l.name())
            for synonym in synonyms:
                synonym_bpe = trainer.task.source_dictionary.encode_line(bpe.encode(synonym)).long()[0:-1]
                untouchable_token_blacklist.extend(synonym_bpe)                

    # overwrite target with user desired output
    samples, target_mask = find_and_replace_target(samples, original_output_token, desired_output_token)
    original_samples = deepcopy(samples)
    original_target = deepcopy(samples['target'])
    num_loops = 1
    if args.get_multiple_results:
        num_loops = 25
    for loop in range(num_loops):
        attack_mode = 'gradient' # gradient or random flipping
        new_found_input_tokens = None
        best_found_loss = 999999999999999        
        changed_positions = [False] * (samples['net_input']['src_tokens'].shape[1] - 1) # if a position is already changed, don't change it again. [False] for the sequence length, but minus -1 to ignore pad
        samples = deepcopy(original_samples)
        for i in range(samples['ntokens'] * 3): # this many iters over a single example. Gradient attack will early stop            
            assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad is always there

            samples, predictions = all_attack_utils.run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=True)
            assert all(torch.eq(samples['target'].squeeze(0), original_target.squeeze(0))) # make sure target is never updated
            if new_found_input_tokens is not None:
                print('\nFinal input', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
                print('Final output', bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(predictions), None)))
                break

            # clear grads, compute new grads, and get candidate tokens            
            candidate_input_tokens = all_attack_utils.get_attack_candidates(trainer, samples, attack_mode, embedding_weight, target_mask=target_mask)

            new_found_input_tokens = None
            batch_size = 64
            all_inference_samples, all_changed_positions = all_attack_utils.build_inference_samples(samples, batch_size, args, candidate_input_tokens, changed_positions, trainer, bpe, untouchable_token_blacklist=untouchable_token_blacklist, adversarial_token_blacklist=adversarial_token_blacklist)

            for inference_indx, inference_sample in enumerate(all_inference_samples):
                predictions = trainer.task.inference_step(generator, [trainer.get_model()],
                    inference_sample) # batched inference
                for prediction_indx, prediction in enumerate(predictions): # for all predictions
                    prediction = prediction[0]['tokens'].cpu()
                    # if prediction is the same, then save input
                    desired_output_token_appeared = False
                    original_output_token_present = False                    

                    if all(token in prediction for token in desired_output_token): # we want the desired_output_token to be present
                        desired_output_token_appeared = True
                    if any(token in prediction for token in original_output_token):  # and we want the original output_token to be gone
                        original_output_token_present = True                        
                    if desired_output_token_appeared and not original_output_token_present:
                        new_found_input_tokens = deepcopy(inference_sample['net_input']['src_tokens'][prediction_indx].unsqueeze(0))
                        changed_positions[all_changed_positions[inference_indx][prediction_indx]] = True
                        break
                    if new_found_input_tokens is not None:
                        break
                if new_found_input_tokens is not None:
                    break
            if new_found_input_tokens is not None:
                samples['net_input']['src_tokens'] = new_found_input_tokens # updating samples doesn't matter because we are done
            else: # get losses and find the best one to keep making progress
                current_best_found_loss = 99999999
                current_best_found_tokens = None
                current_best_found_loss_changed_pos = None
                for inference_indx, inference_sample in enumerate(all_inference_samples):
                    _, __, losses = all_attack_utils.get_input_grad(trainer, inference_sample, target_mask, no_backwards=True, reduce_loss=False)
                    losses = losses.reshape(batch_size, samples['target'].shape[1]) # unflatten losses
                    losses = torch.sum(losses, dim=1) # total loss. Note that for each entry of the batch, all entries are 0 except one.
                    for loss_indx, loss in enumerate(losses):
                        if loss < current_best_found_loss:
                            current_best_found_loss = loss
                            current_best_found_tokens = inference_sample['net_input']['src_tokens'][loss_indx].unsqueeze(0)
                            current_best_found_loss_changed_pos = (inference_indx, loss_indx)

                if current_best_found_loss < best_found_loss: # update best tokens
                    best_found_loss = current_best_found_loss
                    samples['net_input']['src_tokens'] = current_best_found_tokens
                    changed_positions[all_changed_positions[current_best_found_loss_changed_pos[0]][current_best_found_loss_changed_pos[1]]] = True

                # gradient is deterministic, so if it didnt flip another then its never going to
                else:
                    attack_mode = all_attack_utils.update_attack_mode_state_machine(attack_mode)

        # append all of the adversarial tokens we used from last iteration into the source
        for indx, position in enumerate(changed_positions):
            if position:
                adversarial_token_blacklist.append(samples['net_input']['src_tokens'][0][indx].cpu().unsqueeze(0))

    return changed_positions   # return changed_positions for the last iter of num_loops


def malicious_appends(samples, args, trainer, generator, embedding_weight, itr, bpe, i):
    # these ╩ and 上 seem to cause lots of errors for my model but not the target models
    find_ignored_tokens = False # I think the ignored tokens if funny but not that interesting
    if args.interactive_attacks: # get user input and build samples
        samples = attack_utils.get_user_input(trainer, bpe)    
    if samples is None:
        return None
    samples, original_prediction = all_attack_utils.run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=False)
    print(bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
    print(bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(original_prediction), None)))

    adversarial_token_blacklist = []
    adversarial_token_blacklist.extend(trainer.task.source_dictionary.encode_line(bpe.encode('上')).long()[0:-1].cpu())
    adversarial_token_blacklist.extend(trainer.task.source_dictionary.encode_line(bpe.encode('╩')).long()[0:-1].cpu())

    # add random trigger to the user input
    num_loops = 1
    if args.get_multiple_results:
        num_loops = 25
    original_samples = deepcopy(samples)
    for loop in range(num_loops):
        samples = deepcopy(original_samples)

        num_trigger_tokens = 5
        # if punctuation is already present at the end, we want to replace it with a comma. Else, add a comma at the end
        if samples['net_input']['src_tokens'][0][-2] in [5, 129,  88,   4,  89,  43]: # if the token is . ; ! , ? :
            num_tokens_to_add = num_trigger_tokens + 1
        else:
            num_tokens_to_add = num_trigger_tokens + 2 # the extra + 1 is to we can replace the last token with <eos>
        trigger_concatenated_source_tokens = torch.cat((samples['net_input']['src_tokens'][0][0:-1], torch.randint(5, 9, (1, num_tokens_to_add)).cuda().squeeze(0)),dim=0) # the +1th token is replaced by <eos>        
        trigger_concatenated_source_tokens[-num_trigger_tokens - 2] = torch.LongTensor([4]).squeeze(0).cuda() # replace with ,
        trigger_concatenated_source_tokens[-1] = torch.LongTensor([2]).squeeze(0).cuda() # add <eos>
        samples['net_input']['src_tokens'] = trigger_concatenated_source_tokens.unsqueeze(0)
        samples['net_input']['src_lengths'] += num_tokens_to_add - 1

        if samples['target'][0][-2] in [5, 129,  88,   4,  89,  43]:
            samples['target'][0][-2] = torch.LongTensor([4]).squeeze(0).cuda() # replace target punctuation with ,
            samples['net_input']['prev_output_tokens'][0][-1] = torch.LongTensor([4]).squeeze(0).cuda() # replace target punctuation with ,

        original_target = deepcopy(samples['target'])
        attack_mode = 'gradient' # gradient or random flipping
        best_found_loss = 999999999999999
        if not find_ignored_tokens:
            best_found_loss *= -1 # we want small losses for this
        for i in range(samples['ntokens'] * 1): # this many iters over a single example. Gradient attack will early stop
            if i == samples['ntokens'] * 1 - 1:
                print(bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad is always there

            samples, predictions = all_attack_utils.run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=True)
            if i == samples['ntokens'] * 1 - 1:
                print(bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(predictions), None)))                
                continue
            assert all(torch.eq(samples['target'].squeeze(0), original_target.squeeze(0))) # make sure target is never updated
            
            if find_ignored_tokens:
                increase_loss = False
            else:
                increase_loss = True            
            candidate_input_tokens = all_attack_utils.get_attack_candidates(trainer, samples, attack_mode, embedding_weight, target_mask=None, increase_loss=increase_loss)
            candidate_input_tokens = candidate_input_tokens[-num_trigger_tokens:] # the trigger candidates are at the end
            batch_size = 64
            all_inference_samples, _ = all_attack_utils.build_inference_samples(samples, batch_size, args, candidate_input_tokens, None, trainer, bpe, num_trigger_tokens=num_trigger_tokens, adversarial_token_blacklist=adversarial_token_blacklist) # none is to ignore changed_positions

            current_best_found_loss = 9999999
            if not find_ignored_tokens:
                current_best_found_loss *= -1 # we want small losses for this
            current_best_found_tokens = None
            for inference_indx, inference_sample in enumerate(all_inference_samples):
                _, __, losses = all_attack_utils.get_input_grad(trainer, inference_sample, target_mask=None, no_backwards=True, reduce_loss=False)
                losses = losses.reshape(batch_size, samples['target'].shape[1]) # unflatten losses
                losses = torch.sum(losses, dim=1)

                # this subtracts out the loss for copying the input. We don't want copies because they don't transfer. It seems like the black-box systems have heuristics to prevent copying.
                if not find_ignored_tokens:
                    input_sample_target_same_as_source = deepcopy(inference_sample)
                    input_sample_target_same_as_source['target'] = deepcopy(inference_sample['net_input']['src_tokens'])
                    input_sample_target_same_as_source['net_input']['prev_output_tokens'] = torch.cat((input_sample_target_same_as_source['target'][0][-1:], input_sample_target_same_as_source['target'][0][:-1]), dim=0).unsqueeze(dim=0).repeat(batch_size, 1)
                    input_sample_target_same_as_source['ntokens'] = input_sample_target_same_as_source['target'].shape[1] * batch_size
                    _, __, copy_losses = all_attack_utils.get_input_grad(trainer, input_sample_target_same_as_source, target_mask=None, no_backwards=True, reduce_loss=False)
                    copy_losses = copy_losses.reshape(batch_size, input_sample_target_same_as_source['target'].shape[1]) # unflatten losses
                    copy_losses = torch.sum(copy_losses, dim=1)
                    losses = losses + 0.3 * copy_losses

                for loss_indx, loss in enumerate(losses):
                    if find_ignored_tokens:
                        if loss < current_best_found_loss:
                            current_best_found_loss = loss
                            current_best_found_tokens = inference_sample['net_input']['src_tokens'][loss_indx].unsqueeze(0)
                    else:
                        if loss > current_best_found_loss:
                            current_best_found_loss = loss
                            current_best_found_tokens = inference_sample['net_input']['src_tokens'][loss_indx].unsqueeze(0)

            if find_ignored_tokens:
                if current_best_found_loss < best_found_loss: # update best tokens
                    best_found_loss = current_best_found_loss
                    samples['net_input']['src_tokens'] = current_best_found_tokens
                # gradient is deterministic, so if it didnt flip another then its never going to
                else:
                    attack_mode = all_attack_utils.update_attack_mode_state_machine(attack_mode)
            else:
                if current_best_found_loss > best_found_loss: # update best tokens
                    best_found_loss = current_best_found_loss
                    samples['net_input']['src_tokens'] = current_best_found_tokens
                    # gradient is deterministic, so if it didnt flip another then its never going to
                else:
                    attack_mode = all_attack_utils.update_attack_mode_state_machine(attack_mode)

        for indx in range(len(samples['net_input']['src_tokens'][0])):
            adversarial_token_blacklist.append(samples['net_input']['src_tokens'][0][indx].cpu().unsqueeze(0))

if __name__ == '__main__':
    main()