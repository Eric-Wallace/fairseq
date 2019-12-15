import numpy as np
from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer
import nltk
from nltk.corpus import wordnet

extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

# returns the wordpiece embedding weight matrix
def get_embedding_weight(model, bpe_vocab_size):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == bpe_vocab_size:
                return module.weight.detach()
    exit("Embedding matrix not found")

# add hooks for embeddings, only add a hook to encoder wordpiece embeddings (not position)
def add_hooks(model, bpe_vocab_size):
    hook_registered = False
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == bpe_vocab_size:
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)
                hook_registered = True
    if not hook_registered:
        exit("Embedding matrix not found")

# uses gradient-based method for estimating worst/best tokens to replace.
def hotflip_attack(averaged_grad, embedding_matrix, token_ids,
                   increase_loss=False, num_candidates=1):
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    token_embeds = torch.nn.functional.embedding(torch.LongTensor(token_ids),
                                                         embedding_matrix).detach().unsqueeze(0)
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", (averaged_grad, embedding_matrix))
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()

# random search, returns num_candidates random samples.
def random_attack(embedding_matrix, token_ids, num_candidates=1):
    embedding_matrix = embedding_matrix.cpu()
    new_token_ids = [[None]*num_candidates for _ in range(len(token_ids))]
    for token_id in range(len(token_ids)):
        for candidate_number in range(num_candidates):
            # rand token in the embedding matrix
            rand_token = np.random.randint(embedding_matrix.shape[0])
            new_token_ids[token_id][candidate_number] = rand_token
    return new_token_ids

# runs the samples through the model and fills extracted_grads with the gradient w.r.t. the embedding
def get_input_grad(trainer, samples, mask=None, no_backwards=False, reduce_loss=True, eos_loss=False):
    trainer._set_seed()
    trainer.get_model().eval() # we want grads from eval() model, to turn off dropout and stuff
    trainer.criterion.train()
    trainer.zero_grad()

    # fills extracted_grads with the gradient w.r.t. the embedding
    sample = trainer._prepare_sample(samples)
    loss, _, __, prediction = trainer.criterion(trainer.get_model(), sample, return_prediction=True, mask=mask, reduce=reduce_loss, eos_loss=eos_loss)
    if not no_backwards:
        trainer.optimizer.backward(loss)
    return sample['net_input']['src_lengths'], prediction.max(2)[1].squeeze().detach().cpu(), loss.detach().cpu()

# run model to get predictions and save those predictions into the targets for samples
def run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=False):
    if torch.cuda.is_available() and not args.cpu:
        samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
        samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()
        if 'target' in samples:
            samples['target'] = samples['target'].cuda()
            samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()

    translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
    predictions = translations[0][0]['tokens'].cpu()
    if no_overwrite:
        return samples, predictions

    samples['target'] = translations[0][0]['tokens'].unsqueeze(dim=0)
    # prev_output_tokens is the right rotated version of the target
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:], samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)

    return samples, predictions

# find the position of the start and end of the original_output_token and replaces it with desired_output_token
# desired_output_token can be shorter, longer, or the same length as original_output_token
def find_and_replace_target(samples, original_output_token, desired_output_token):
    mask = []
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
    mask = [0] * start_pos + [1] * len(desired_output_token) + [0] * (len(new_target) - len(desired_output_token) - start_pos)
    samples['target'] = new_target.unsqueeze(0)
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:], samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)
    return samples, mask


# take samples (of batch size 1) and repeat it batch_size times to do batched inference / loss calculation
def build_inference_samples(samples, batch_size, args, candidate_input_tokens, changed_positions, trainer, bpe, untouchable_token_blacklist=None, adversarial_token_blacklist=None, num_trigger_tokens=None):
    # move to cpu to make this faster
    samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cpu()
    samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cpu()
    samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cpu()
    samples['target'] = samples['target'].cpu()

    # copy and repeat the samples instead batch size elements
    samples_repeated_by_batch = deepcopy(samples)
    samples_repeated_by_batch['ntokens'] *= batch_size
    samples_repeated_by_batch['target'] = samples_repeated_by_batch['target'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['prev_output_tokens'] = samples_repeated_by_batch['net_input']['prev_output_tokens'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['src_tokens'] = samples_repeated_by_batch['net_input']['src_tokens'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['src_lengths'] = samples_repeated_by_batch['net_input']['src_lengths'].repeat(batch_size, 1)
    samples_repeated_by_batch['nsentences'] = batch_size

    all_inference_samples = [] # stores a list of batches of candidates
    all_changed_positions = [] # stores all the changed_positions for each batch element
    current_batch_size = 0
    current_batch_changed_position = []
    current_inference_samples = deepcopy(samples_repeated_by_batch) # stores one batch worth of candidates
    for index in range(len(candidate_input_tokens)): # for all the positions in the input
        for token_id in candidate_input_tokens[index]: # for all the candidates
            if num_trigger_tokens is not None: # want to change the last tokens, not the first, for triggers
                index_to_use = index - num_trigger_tokens - 1 # -1 to skip <eos>
            else:
                index_to_use = index

            if untouchable_token_blacklist is not None and current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use].cpu() in untouchable_token_blacklist: # don't touch the word if its in the blacklist
                continue
            if adversarial_token_blacklist is not None and any([token_id == blacklisted_token for blacklisted_token in adversarial_token_blacklist]): # don't insert any blacklisted tokens into the source side
                continue
            if args.malicious_nonsense and changed_positions[index]: # if we have already changed this position, skip. Only do this for malicious nonsense
                continue

            original_token = deepcopy(current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use])
            current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] = torch.LongTensor([token_id]).squeeze(0) # change onetoken

            # check if the BPE has changed, and if so, replace the samples
            if not args.no_check_resegmentation:
                string_input_tokens = bpe.decode(trainer.task.source_dictionary.string(current_inference_samples['net_input']['src_tokens'][current_batch_size].cpu(), None))
                retokenized_string_input_tokens = trainer.task.source_dictionary.encode_line(bpe.encode(string_input_tokens)).long().unsqueeze(dim=0)
                #print(bpe.decode(trainer.task.source_dictionary.string(retokenized_string_input_tokens[0].cpu(), None)))
                #print(bpe.decode(trainer.task.source_dictionary.string(current_inference_samples['net_input']['src_tokens'][current_batch_size].cpu(), None)))
                #print(retokenized_string_input_tokens[0].cpu())
                #print(current_inference_samples['net_input']['src_tokens'][current_batch_size])
                #dasdasd = input('enter')
                if len(retokenized_string_input_tokens[0]) != len(current_inference_samples['net_input']['src_tokens'][current_batch_size]) or not torch.all(torch.eq(retokenized_string_input_tokens[0],current_inference_samples['net_input']['src_tokens'][current_batch_size])):
                    current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] = original_token
                    continue

            current_batch_size += 1
            current_batch_changed_position.append(index_to_use) # save its changed position

            if current_batch_size == batch_size: # batch is full
                all_inference_samples.append(deepcopy(current_inference_samples))
                current_inference_samples = deepcopy(samples_repeated_by_batch)
                current_batch_size = 0
                all_changed_positions.append(current_batch_changed_position)
                current_batch_changed_position = []

    samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
    samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()
    samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()
    samples['target'] = samples['target'].cuda()
    for item in all_inference_samples:
        item['net_input']['src_tokens'] = item['net_input']['src_tokens'].cuda()
        item['net_input']['src_lengths'] = item['net_input']['src_lengths'].cuda()
        item['net_input']['prev_output_tokens'] = item['net_input']['prev_output_tokens'].cuda()
        item['target'] = item['target'].cuda()
    return all_inference_samples, all_changed_positions

def get_attack_candidates(trainer, samples, attack_mode, embedding_weight, mask=None, eos_loss=False, increase_loss=False):
    src_lengths, _, __ = get_input_grad(trainer, samples, mask=mask, eos_loss=eos_loss) # gradient is now filled
    if 'gradient' in attack_mode:
        if len(extracted_grads) > 1: # this is for models with shared embedding
            # position [1] in extracted_grads is the encoder embedding grads, [0] is decoder
            if attack_mode == 'gradient':
                gradient_position = 1
            elif attack_mode == 'decoder_gradient':
                gradient_position = 0
        else:
            gradient_position = 0
        input_gradient = extracted_grads[gradient_position][0][0:src_lengths[0]-1] # first [] gets decoder/encoder grads, then batch (currently size 1), then we index into before the padding (though there shouldn't be any padding at the moment because batch size 1). The -1 is to ignore the padding
        candidate_input_tokens = hotflip_attack(input_gradient,
                                                  embedding_weight,
                                                  samples['net_input']['src_tokens'].cpu().numpy()[0],
                                                  num_candidates=50,
                                                  increase_loss=increase_loss)
    elif attack_mode == 'random':
        candidate_input_tokens = random_attack(embedding_weight,
                                                 samples['net_input']['src_tokens'].cpu().numpy()[0],
                                                 num_candidates=50)
        candidate_input_tokens = candidate_input_tokens[0:-1] # remove the candidate for the padding token


    return candidate_input_tokens


def read_file_input(trainer, bpe, i):
    data = open('subsample_test','r').readlines()
    if i < len(data):
        file_input = data[i]
    else:
        return "invalid"
    if '\'' in file_input or '\"' in file_input or '\\' in file_input or file_input == '\n' or file_input.strip() == '' or len(file_input) > 60:
        return "invalid"
    # tokenize input and get lengths
    tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(file_input)).long().unsqueeze(dim=0)
    length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])])
    # build samples and set their targets with the model predictions
    samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])} 
    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    for token in tokenized_bpe_input[0]:
        if torch.eq(token, bpe_vocab_size) or torch.gt(token, bpe_vocab_size): # UNK
            return 'invalid'
    return samples

def get_user_input(trainer, bpe):
    user_input = input('Enter your sentence: ')
    # tokenize input and get lengths
    tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(user_input)).long().unsqueeze(dim=0)
    length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])])
    # build samples and set their targets with the model predictions
    samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])}
    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    for token in tokenized_bpe_input[0]:
        if torch.eq(token, bpe_vocab_size) or torch.gt(token, bpe_vocab_size): # UNK
            return 'invalid' 
    return samples

def update_attack_mode_state_machine(attack_mode):
    if attack_mode == 'gradient': # once gradient fails, start using the decoder gradient
        attack_mode = 'decoder_gradient'
        #print('no more succesful flips, switching from gradient to decoder_gradient')
    elif attack_mode == 'decoder_gradient':
        attack_mode = 'random'
        #print('no more succesful flips, switching from decoder_gradient to random')
    return attack_mode

def main(args):
    utils.import_user_module(args)
    args.max_sentences_valid = 1  # batch size 1 at the moment

    # Initialize CUDA
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # setup task, model, loss function, and trainer
    task = tasks.setup_task(args)
    if not args.interactive_attacks and not args.read_file_input:
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
    add_hooks(trainer.get_model(), bpe_vocab_size) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(trainer.get_model(), bpe_vocab_size) # save the embedding matrix
    if not args.interactive_attacks and not args.read_file_input:
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
        elif args.malicious_nonsense:
            if args.random_start:
                changed_positions = random_start_malicious_nonsense(samples, args, trainer, generator, embedding_weight, itr, bpe, i)
            else:
                changed_positions = malicious_nonsense(samples, args, trainer, generator, embedding_weight, itr, bpe, i)
        elif args.malicious_appends:
            malicious_appends(samples, args, trainer, generator, embedding_weight, itr, bpe, i)
        else:
            exit("pick an attack mode using --targeted-flips, --malicious-nonsense, --malicious-appends, or --universal-triggers")
        if changed_positions is None:
            #print('error, skipping example')
            continue

        num_total_samples += 1.0
        # print(changed_positions)
        if any(changed_positions):
            num_samples_changed += 1.0
            num_tokens_changed += sum(changed_positions)
            total_num_tokens += len(changed_positions)
        # print('\n')
    if num_total_samples > 0.0:
         print('\n\n\n')
         print('Total Num Samples', num_total_samples)
         print('Percent Samples Changed', num_samples_changed / num_total_samples)
         print('Percent Tokens Changed', num_tokens_changed / total_num_tokens)


def malicious_nonsense(samples, args, trainer, generator, embedding_weight, itr, bpe, i):
    if args.interactive_attacks: # get user input and build samples
        samples = get_user_input(trainer, bpe)
    elif args.read_file_input:
        samples = read_file_input(trainer, bpe, i)
    if samples == 'invalid':
        return None

    samples, original_prediction = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=False)
    original_samples = deepcopy(samples)
    adversarial_token_blacklist = []
    if args.get_multiple_results:
        num_loops = 25
    else:
        num_loops = 1
    for loop in range(num_loops):
        attack_mode = 'gradient' # gradient or random flipping
        new_found_input_tokens = 'temp' # for the first very iteration, we want to print so we set this to something that isn't None

        changed_positions = [False] * (samples['net_input']['src_tokens'].shape[1] - 1) # if a position is already changed, don't change it again. [False] for the sequence length, but minus -1 to ignore pad
        samples = deepcopy(original_samples)        
        for i in range(samples['ntokens'] * 3 + 1): # this many iters over a single example. Gradient attack will early stop
            #if new_found_input_tokens is not None: # only print when a new best has been found
                #print(bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad it always there

            samples, predictions = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=True)
            # clear grads, compute new grads, and get candidate tokens
            global extracted_grads
            extracted_grads = []
            candidate_input_tokens = get_attack_candidates(trainer, samples, attack_mode, embedding_weight, mask=None)

            new_found_input_tokens = None
            batch_size = 64
            all_inference_samples, all_changed_positions = build_inference_samples(samples, batch_size, args, candidate_input_tokens, changed_positions, trainer, bpe, adversarial_token_blacklist=adversarial_token_blacklist)

            for inference_indx, inference_sample in enumerate(all_inference_samples):
                predictions = trainer.task.inference_step(generator, [trainer.get_model()],
                    inference_sample) # batched inference
                for prediction_indx, prediction in enumerate(predictions): # for all predictions
                    prediction = prediction[0]['tokens'].cpu()
                    # if prediction is the same, then save input
                    if prediction.shape == original_prediction.shape and all(torch.eq(prediction,original_prediction)):
                        if all(torch.eq(inference_sample['net_input']['src_tokens'][prediction_indx],samples['net_input']['src_tokens'].squeeze(0))): # if the "new" candidate is actually the same as the current tokens
                            continue
                        new_found_input_tokens = deepcopy(inference_sample['net_input']['src_tokens'][prediction_indx].unsqueeze(0))
                        changed_positions[all_changed_positions[inference_indx][prediction_indx]] = True
                        break # break twice
                if new_found_input_tokens is not None:
                    break

            # Update current input if the new candidate flipped a position
            if new_found_input_tokens is not None:
                if attack_mode == 'random':
                    attack_mode = 'gradient'
                    #print('random worked, switching back to gradient')
                samples['net_input']['src_tokens'] = new_found_input_tokens

            # gradient is deterministic, so if it didnt flip another then its never going to
            else:
                attack_mode = update_attack_mode_state_machine(attack_mode)
                if attack_mode == "random": # TODO, just to speed things up have this break
                    break

        for indx, position in enumerate(changed_positions):
            if position:
                adversarial_token_blacklist.append(samples['net_input']['src_tokens'][0][indx].cpu().unsqueeze(0))

        if any(changed_positions):
            print(bpe.decode(trainer.task.source_dictionary.string(original_samples['net_input']['src_tokens'].cpu()[0], None)))
            print(bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))

    return changed_positions


def random_start_malicious_nonsense(samples, args, trainer, generator, embedding_weight, itr, bpe, i):
    attack_mode = 'gradient' # gradient or random flipping
    new_found_input_tokens = None
    best_found_loss = 999999999999999
    if args.interactive_attacks: # get user input and build samples
        samples = get_user_input(trainer, bpe)
    elif args.read_file_input:
        samples = read_file_input(trainer, bpe, i)
    if samples == 'invalid':
        return None

    samples, original_prediction = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=False)
    changed_positions = [False] * (samples['net_input']['src_tokens'].shape[1] - 1) # if a position is already changed, don't change it again. [False] for the sequence length, but minus -1 to ignore pad
    samples['net_input']['src_tokens'] = torch.randint(3, trainer.get_model().encoder.embed_tokens.weight.shape[0], samples['net_input']['src_tokens'].shape).cuda() # TODO, I think start with a 3. I want to avoid <bos> and stuff
    samples['net_input']['src_tokens'][0][-1] = torch.LongTensor([2]).squeeze(0).cuda() # add <eos>
    for i in range(samples['ntokens'] * 3): # this many iters over a single example. Gradient attack will early stop
        print('\nCurrent Input  ', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
        assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad it always there
        samples, predictions = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=True)
        if new_found_input_tokens is not None:
            print('Final Input ', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            print('Final Output ', bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(predictions), None)))
            break
        print('Current Target   ', bpe.decode(trainer.task.target_dictionary.string(samples['target'].cpu()[0], None)))
        print('Current Predict  ', bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(predictions), None)))

        # clear grads, compute new grads, and get candidate tokens
        global extracted_grads
        extracted_grads = []
        candidate_input_tokens = get_attack_candidates(trainer, samples, attack_mode, embedding_weight, mask=None)

        new_found_input_tokens = None
        batch_size = 64
        all_inference_samples, all_changed_positions = build_inference_samples(samples, batch_size, args, candidate_input_tokens, changed_positions, trainer, bpe)

        if best_found_loss < 8.0:
            for inference_indx, inference_sample in enumerate(all_inference_samples):
                predictions = trainer.task.inference_step(generator, [trainer.get_model()],inference_sample) # batched inference
                for prediction_indx, prediction in enumerate(predictions): # for all predictions
                    prediction = prediction[0]['tokens'].cpu()
                    # if prediction is the same, then save input
                    if prediction.shape == original_prediction.shape and all(torch.eq(prediction,original_prediction)):
                        new_found_input_tokens = deepcopy(inference_sample['net_input']['src_tokens'][prediction_indx].unsqueeze(0))
                        break
                if new_found_input_tokens is not None:
                    break
            if new_found_input_tokens is not None:
                samples['net_input']['src_tokens'] = new_found_input_tokens # updating samples doesn't matter because we are done

        else: # get losses and find the best one to keep making progress
            current_best_found_loss = 99999999
            current_best_found_tokens = None
            current_best_found_loss_changed_pos = None
            found_it = False
            for inference_indx, inference_sample in enumerate(all_inference_samples):
                _, __, losses = get_input_grad(trainer, inference_sample, mask=None, no_backwards=True, reduce_loss=False)
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

            # gradient is deterministic, so if it didnt flip another then its never going to
            else:
                attack_mode = update_attack_mode_state_machine(attack_mode)

    return None #changed_positions # TODO, what is the metric here? because we don't really care about changed_positions


def targeted_flips(samples, args, trainer, generator, embedding_weight, itr, bpe, i):
    assert args.interactive_attacks or args.read_file_input # only interactive for now
    if args.interactive_attacks: # get user input and build samples
        samples = get_user_input(trainer, bpe)
    elif args.read_file_input:
        samples = read_file_input(trainer, bpe, i)
    if samples == 'invalid':
        return None

    samples, original_prediction = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=False)
    if args.interactive_attacks:
        print('Current Translation ', bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(original_prediction), None)))
        original_output_token = input('Enter the target token ')
        desired_output_token = input('Enter desired target token ')
        adversarial_token_blacklist_string = input('Enter optional space seperated blacklist of invalid adversarial words ')
        untouchable_token_blacklist_string = input('Enter optional space seperated blacklist of source words to keep ')

        # -1 strips off <eos> token
        original_output_token = trainer.task.target_dictionary.encode_line(bpe.encode(original_output_token)).long()[0:-1]
        desired_output_token = trainer.task.target_dictionary.encode_line(bpe.encode(desired_output_token)).long()[0:-1]
        # if len(original_output_token) != 1 or len(desired_output_token) != 1:
        #    print("Error: more than one BPE token", len(original_output_token), len(desired_output_token))
        #    return
        print("Original Output Len", len(original_output_token), "Desired Output Len", len(desired_output_token))
        #    return

        # don't change any of these tokens in the input
        untouchable_token_blacklist = []
        if untouchable_token_blacklist_string is not None and untouchable_token_blacklist_string != '' and untouchable_token_blacklist_string != '\n':
            untouchable_token_blacklist_string = untouchable_token_blacklist_string.split(' ')
            for token in untouchable_token_blacklist_string:
                token = trainer.task.source_dictionary.encode_line(bpe.encode(token)).long()[0:-1]
                #if len(token) == 1:
                #    untouchable_token_blacklist.append(token.squeeze(0))
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
                # if len(synonym_bpe) == 1:
                #     adversarial_token_blacklist.append(synonym_bpe.squeeze(0))

    # overwrite target with user desired output
    samples, mask = find_and_replace_target(samples, original_output_token, desired_output_token)
    original_samples = deepcopy(samples)
    original_target = deepcopy(samples['target'])
    num_loops = 1
    if args.get_multiple_results:
        num_loops = 25
    for loop in range(num_loops):
        attack_mode = 'gradient' # gradient or random flipping
        new_found_input_tokens = None
        best_found_loss = 999999999999999
        # print("Untouchable Tokens: ", [bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(token.unsqueeze(0)), None)) for token in untouchable_token_blacklist])
        # print("Blacklisted Adv. Tokens: ", [bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(token.unsqueeze(0)), None)) for token in adversarial_token_blacklist])
        changed_positions = [False] * (samples['net_input']['src_tokens'].shape[1] - 1) # if a position is already changed, don't change it again. [False] for the sequence length, but minus -1 to ignore pad
        samples = deepcopy(original_samples)
        for i in range(samples['ntokens'] * 3): # this many iters over a single example. Gradient attack will early stop
            # print('\nCurrent Input ', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad is always there

            samples, predictions = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=True)
            # print('Current Output ', bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(predictions), None)))
            assert all(torch.eq(samples['target'].squeeze(0), original_target.squeeze(0))) # make sure target is never updated
            if new_found_input_tokens is not None:
                print('\nFinal input', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
                print('Final output', bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(predictions), None)))
                break

            # clear grads, compute new grads, and get candidate tokens
            global extracted_grads
            extracted_grads = [] # clear old extracted_grads
            candidate_input_tokens = get_attack_candidates(trainer, samples, attack_mode, embedding_weight, mask=mask)

            new_found_input_tokens = None
            batch_size = 64
            all_inference_samples, all_changed_positions = build_inference_samples(samples, batch_size, args, candidate_input_tokens, changed_positions, trainer, bpe, untouchable_token_blacklist=untouchable_token_blacklist, adversarial_token_blacklist=adversarial_token_blacklist)

            for inference_indx, inference_sample in enumerate(all_inference_samples):
                predictions = trainer.task.inference_step(generator, [trainer.get_model()],
                    inference_sample) # batched inference
                for prediction_indx, prediction in enumerate(predictions): # for all predictions
                    prediction = prediction[0]['tokens'].cpu()
                    # if prediction is the same, then save input
                    desired_output_token_appeared = False
                    original_output_token_present = False
                    # for idx, current_token in enumerate(prediction):

                    if all(token in prediction for token in desired_output_token): # we want the desired_output_token to be present
                        desired_output_token_appeared = True
                    if any(token in prediction for token in original_output_token):  # and we want the original output_token to be gone
                        original_output_token_present = True
                        # if current_token in desired_output_token: # we want the desired_output_token to be present
                        #     desired_output_token_appeared = True
                        # elif current_token == original_output_token: # and we want the original output_token to be gone
                        #     original_output_token_present = True
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
                    _, __, losses = get_input_grad(trainer, inference_sample, mask, no_backwards=True, reduce_loss=False)
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
                    attack_mode = update_attack_mode_state_machine(attack_mode)

        # append all of the adversarial tokens we used from last iteration into the source
        for indx, position in enumerate(changed_positions):
            if position:
                adversarial_token_blacklist.append(samples['net_input']['src_tokens'][0][indx].cpu().unsqueeze(0))

    return changed_positions   # return changed_positions for the last iter of num_loops


def malicious_appends(samples, args, trainer, generator, embedding_weight, itr, bpe, i):
    # these ╩ and 上 seem to cause lots of errors for my model but not the target models
    find_ignored_tokens = False # I think the ignored tokens if funny but not that interesting
    if args.interactive_attacks: # get user input and build samples
        samples = get_user_input(trainer, bpe)
    elif args.read_file_input:
        samples = read_file_input(trainer, bpe, i)
    if samples == 'invalid':
        return None
    samples, original_prediction = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=False)
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
        #trigger_concatenated_source_tokens = torch.cat((samples['net_input']['src_tokens'][0][0:-1], torch.randint(3, trainer.get_model().encoder.embed_tokens.weight.shape[0], (1, num_tokens_to_add)).cuda().squeeze(0)),dim=0) # the +1th token is replaced by <eos>
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

            samples, predictions = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=True)
            if i == samples['ntokens'] * 1 - 1:
                print(bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(predictions), None)))
                #print()
                continue
            assert all(torch.eq(samples['target'].squeeze(0), original_target.squeeze(0))) # make sure target is never updated

            # clear grads, compute new grads, and get candidate tokens
            global extracted_grads
            extracted_grads = [] # clear old extracted_grads
            if find_ignored_tokens:
                increase_loss = False
            else:
                increase_loss = True
            eos_loss = False # TODO, this doesn't seem to work when turned on
            candidate_input_tokens = get_attack_candidates(trainer, samples, attack_mode, embedding_weight, mask=None, increase_loss=increase_loss, eos_loss=eos_loss)
            candidate_input_tokens = candidate_input_tokens[-num_trigger_tokens:] # the trigger candidates are at the end
            batch_size = 64
            all_inference_samples, _ = build_inference_samples(samples, batch_size, args, candidate_input_tokens, None, trainer, bpe, num_trigger_tokens=num_trigger_tokens, adversarial_token_blacklist=adversarial_token_blacklist) # none is to ignore changed_positions

            current_best_found_loss = 9999999
            if not find_ignored_tokens:
                current_best_found_loss *= -1 # we want small losses for this
            current_best_found_tokens = None
            for inference_indx, inference_sample in enumerate(all_inference_samples):
                _, __, losses = get_input_grad(trainer, inference_sample, mask=None, no_backwards=True, reduce_loss=False, eos_loss=eos_loss)
                losses = losses.reshape(batch_size, samples['target'].shape[1]) # unflatten losses
                losses = torch.sum(losses, dim=1)

                # this subtracts out the loss for copying the input. We don't want copies because they don't transfer. It seems like the black-box systems have heuristics to prevent copying.
                if not find_ignored_tokens:
                    input_sample_target_same_as_source = deepcopy(inference_sample)
                    input_sample_target_same_as_source['target'] = deepcopy(inference_sample['net_input']['src_tokens'])
                    input_sample_target_same_as_source['net_input']['prev_output_tokens'] = torch.cat((input_sample_target_same_as_source['target'][0][-1:], input_sample_target_same_as_source['target'][0][:-1]), dim=0).unsqueeze(dim=0).repeat(batch_size, 1)
                    input_sample_target_same_as_source['ntokens'] = input_sample_target_same_as_source['target'].shape[1] * batch_size
                    _, __, copy_losses = get_input_grad(trainer, input_sample_target_same_as_source, mask=None, no_backwards=True, reduce_loss=False, eos_loss=eos_loss)
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
                    attack_mode = update_attack_mode_state_machine(attack_mode)
            else:
                if current_best_found_loss > best_found_loss: # update best tokens
                    best_found_loss = current_best_found_loss
                    samples['net_input']['src_tokens'] = current_best_found_tokens
                    # gradient is deterministic, so if it didnt flip another then its never going to
                else:
                    attack_mode = update_attack_mode_state_machine(attack_mode)

        for indx in range(len(samples['net_input']['src_tokens'][0])):
            adversarial_token_blacklist.append(samples['net_input']['src_tokens'][0][indx].cpu().unsqueeze(0))

parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
# make sure everything is reset before loading the model
args.reset_optimizer = True
args.reset_meters = True
args.reset_dataloader = True
args.reset_lr_scheduler = True
args.path = args.restore_file
main(args)

