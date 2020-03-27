import numpy as np
from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer
import attack_utils
import all_attack_utils

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
    for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
        universal_attack(samples, args, trainer, generator, embedding_weight, itr, bpe, i)
        TODO, add flag for the two different types of attacks

def universal_attack(samples, args, trainer, generator, embedding_weight, itr, bpe, i):
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