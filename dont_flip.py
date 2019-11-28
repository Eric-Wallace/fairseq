import numpy as np
from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer

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
def get_input_grad(trainer, samples, mask=None):
    trainer._set_seed()
    trainer.get_model().eval() # we want grads from eval() model, to turn off dropout and stuff
    trainer.criterion.train()
    trainer.zero_grad()

    # fills extracted_grads with the gradient w.r.t. the embedding
    sample = trainer._prepare_sample(samples)
    loss, _, __, prediction = trainer.criterion(trainer.get_model(), sample, return_prediction=True, mask=mask)
    trainer.optimizer.backward(loss)
    return sample['net_input']['src_lengths'], prediction.max(2)[1].squeeze().detach().cpu()

def main(args):
    utils.import_user_module(args)
    args.max_sentences_valid = 1  # batch size 1 at the moment

    # Initialize CUDA
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
    print(args); print(task); print(model); print(criterion); print(generator)
    if args.targeted_flips:
        targeted_flips(args, trainer, generator)
    else:
        malicious_nonsense(args, trainer, generator)

def malicious_nonsense(args, trainer, generator):
    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    add_hooks(trainer.get_model(), bpe_vocab_size) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(trainer.get_model(), bpe_vocab_size) # save the embedding matrix

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

    # run model to get predictions and save those predictions into the targets for samples
    def run_inference_and_overwrite_samples(samples):
        if torch.cuda.is_available() and not args.cpu:
            samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
            samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()
            if 'target' in samples:
                samples['target'] = samples['target'].cuda()
                samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()
        translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
        samples['target'] = translations[0][0]['tokens'].unsqueeze(dim=0)
        # prev_output_tokens is the right rotated version of the target
        samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:], samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)        
        predictions = translations[0][0]['tokens'].cpu()
        return samples, predictions

    num_samples_changed = 0.0
    num_total_samples = 0.0
    num_tokens_changed = 0.0
    total_num_tokens = 0.0
    for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
        attack_mode = 'decoder_gradient' # gradient or random flipping
        new_found_input_tokens = 'temp' # for the first very iteration, we want to print so we set this to something that isn't None
        if args.interactive_attacks: # get user input and build samples
            user_input = input('Enter your sentence:\n')
            # tokenize input and get lengths
            tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(user_input)).long().unsqueeze(dim=0)
            length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])])
            # build samples and set their targets with the model predictions
            samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])}
                
        samples, original_prediction = run_inference_and_overwrite_samples(samples)        
        changed_positions = [False] * (samples['net_input']['src_tokens'].shape[1] - 1) # if a position is already changed, don't change it again. [False] for the sequence length, but minus -1 to ignore pad
        # if args.random_start:        
        #     samples['net_input']['src_tokens'] = torch.randint(2, bpe_vocab_size, samples['net_input']['src_tokens'].shape).cuda() # TODO, I think start a 2? I want to avoid <bos> and stuff            
        for i in range(samples['ntokens'] * 3): # this many iters over a single batch. Gradient attack will early stop
            if new_found_input_tokens is not None: # only print when a new best has been found
                print(bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad it always there

            samples, predictions = run_inference_and_overwrite_samples(samples)
            #print(bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(predictions), None)))
            global extracted_grads
            extracted_grads = [] # clear old extracted_grads
            src_lengths, _ = get_input_grad(trainer, samples) # gradient is now filled
            if 'gradient' in attack_mode:
                # position [1] in extracted_grads is the encoder embedding grads, [0] is decoder
                if attack_mode == 'gradient':
                    gradient_position = 1
                elif attack_mode == 'decoder_gradient':
                    gradient_position = 0
                input_gradient = extracted_grads[gradient_position][0][0:src_lengths[0]-1] # first [] gets decoder/encoder grads, then batch (currently size 1), then we index into before the padding (though there shouldn't be any padding at the moment because batch size 1). The -1 is to ignore the padding
                candidate_input_tokens = hotflip_attack(input_gradient,
                                                          embedding_weight,
                                                          samples['net_input']['src_tokens'].cpu().numpy()[0],
                                                          num_candidates=200,
                                                          increase_loss=False)
            elif attack_mode == 'random':
                candidate_input_tokens = random_attack(embedding_weight,
                                                         samples['net_input']['src_tokens'].cpu().numpy()[0],
                                                         num_candidates=200)

            new_found_input_tokens = None
            batch_size = 64
            # take samples (of batch size 1) and repeat it batch_size times
            samples_repeated_by_batch = deepcopy(samples)
            samples_repeated_by_batch['ntokens'] *= batch_size
            samples_repeated_by_batch['target'] = samples_repeated_by_batch['target'].repeat(batch_size, 1)
            samples_repeated_by_batch['net_input']['prev_output_tokens'] = samples_repeated_by_batch['net_input']['prev_output_tokens'].repeat(batch_size, 1)
            samples_repeated_by_batch['net_input']['src_tokens'] = samples_repeated_by_batch['net_input']['src_tokens'].repeat(batch_size, 1)
            samples_repeated_by_batch['net_input']['src_lengths'] = samples_repeated_by_batch['net_input']['src_lengths'].repeat(batch_size, 1)

            all_inference_samples = [] # stores a list of batches of candidates
            current_batch_size = 0
            all_changed_positions = [] # stores all the changed_positions for each batch element
            current_batch_changed_position = []
            current_inference_samples = deepcopy(samples_repeated_by_batch) # stores one batch worth of candidates
            for index in range(len(candidate_input_tokens)): # for all the positions in the input
                for token_id in candidate_input_tokens[index]: # for all the candidates                    
                    if changed_positions[index]: # if we have already changed this position, skip
                        continue

                    current_inference_samples['net_input']['src_tokens'][current_batch_size][index] = torch.LongTensor([token_id]).cuda().squeeze(0) # change on token                    
                    current_batch_size += 1
                    current_batch_changed_position.append(index) # save its changed position

                    if current_batch_size == batch_size: # batch is full
                        all_inference_samples.append(deepcopy(current_inference_samples))
                        current_inference_samples = deepcopy(samples_repeated_by_batch)
                        current_batch_size = 0
                        all_changed_positions.append(current_batch_changed_position)
                        current_batch_changed_position = []

            for inference_indx, inference_sample in enumerate(all_inference_samples):
                predictions = trainer.task.inference_step(generator, [trainer.get_model()],
                    inference_sample) # batched inference
                for prediction_indx, prediction in enumerate(predictions): # for all predictions
                    prediction = prediction[0]['tokens'].cpu()
                    # if prediction is the same, then save input
                    if prediction.shape == original_prediction.shape and all(torch.eq(prediction,original_prediction)):
                        if all(torch.eq(inference_sample['net_input']['src_tokens'][prediction_indx],samples['net_input']['src_tokens'].squeeze(0))): # if the "new" candidate is actually the same as the current tokens
                            print('lol it was the same')
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
                if attack_mode == 'gradient': # once gradient fails, start using the decoder gradient
                    attack_mode = 'decoder_gradient'
                    #print('no more succesful flips, switching from gradient to decoder_gradient')
                elif attack_mode == 'decoder_gradient':
                    attack_mode = 'random'
                    break
                    #print('no more succesful flips, switching from decoder_gradient to random')
        num_total_samples += 1.0
        print(changed_positions)
        if any(changed_positions):
            num_samples_changed += 1.0
            num_tokens_changed += sum(changed_positions)
            total_num_tokens += len(changed_positions)
        print('\n')
    print('Total Num Samples', num_total_samples)
    print('Percent Samples Changed', num_samples_changed / num_total_samples)
    print('Percent Tokens Changed', num_tokens_changed / total_num_tokens)



def targeted_flips(args, trainer, generator):
    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    add_hooks(trainer.get_model(), bpe_vocab_size) # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(trainer.get_model(), bpe_vocab_size) # save the embedding matrix

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

    # run model to get predictions and save those predictions into the targets for samples
    def run_inference_and_overwrite_samples(samples):
        if torch.cuda.is_available() and not args.cpu:
            samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
            samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()
            if 'target' in samples:
                samples['target'] = samples['target'].cuda()
                samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()
        translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
        samples['target'] = translations[0][0]['tokens'].unsqueeze(dim=0)
        # prev_output_tokens is the right rotated version of the target
        samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:], samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)        
        predictions = translations[0][0]['tokens'].cpu()
        return samples, predictions

    num_samples_changed = 0.0
    num_total_samples = 0.0
    num_tokens_changed = 0.0
    total_num_tokens = 0.0
    for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
        attack_mode = 'gradient' # gradient or random flipping
        new_found_input_tokens = 'temp' # for the first very iteration, we want to print so we set this to something that isn't None
        if args.interactive_attacks: # get user input and build samples
            user_input = input('Enter your sentence:\n')
            # tokenize input and get lengths
            tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(user_input)).long().unsqueeze(dim=0)
            length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])])
            # build samples and set their targets with the model predictions
            samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])}
                
        samples, original_prediction = run_inference_and_overwrite_samples(samples)        
        changed_positions = [False] * (samples['net_input']['src_tokens'].shape[1] - 1) # if a position is already changed, don't change it again. [False] for the sequence length, but minus -1 to ignore pad
        if args.interactive_attacks:
            print(bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            print(bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(original_prediction), None)))
            print(samples['net_input']['src_tokens'])
            print(samples['target'])
            source_position = int(input('Enter the source position of the word'))
            target_position = int(input('Enter the target position of the word'))
            original_output_token = samples['target'].cpu()[0][target_position]
            print(bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0][source_position].unsqueeze(0), None)))
            print(bpe.decode(trainer.task.target_dictionary.string(samples['target'].cpu()[0][target_position].unsqueeze(0), None)))
            desired_output_token = input('Enter desired target token')
            desired_output_token = trainer.task.source_dictionary.encode_line(bpe.encode(desired_output_token)).long()[0:-1] # -1 strips off <eos> token (id 2)
            #print(tokenized_target_token)
            #print(bpe.decode(trainer.task.source_dictionary.string(tokenized_target_token.unsqueeze(0), None)))
            print('\n\n')
            if len(desired_output_token) != 1:
                print("NOT ONE BPE TOKEN!!!")
                continue

        for i in range(samples['ntokens'] * 3): # this many iters over a single batch. Gradient attack will early stop
            print('\nCurrent Input  ', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad it always there

            samples, predictions = run_inference_and_overwrite_samples(samples)
            print('Current Output ', bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(predictions), None)))
            def find_and_replace_target(samples, original_output_token, desired_output_token):
                mask = []
                for idx, current_token in enumerate(samples['target'].cpu()[0]):
                    if current_token == original_output_token:
                        mask.append(1)
                        samples['target'][0][idx] = desired_output_token
                    else:
                        mask.append(0)
                        #samples['target'][0][idx] = torch.LongTensor([2]).cuda().squeeze(0)
                return samples, mask
            samples, mask = find_and_replace_target(samples, original_output_token, desired_output_token)
            print('Current Target ', samples['target'])
            print('Current Target ', bpe.decode(trainer.task.source_dictionary.string(samples['target'], None)))
            global extracted_grads
            extracted_grads = [] # clear old extracted_grads
            src_lengths, _ = get_input_grad(trainer, samples, mask) # gradient is now filled
            if 'gradient' in attack_mode:
                # position [1] in extracted_grads is the encoder embedding grads, [0] is decoder
                if attack_mode == 'gradient':
                    gradient_position = 1
                elif attack_mode == 'decoder_gradient':
                    gradient_position = 0
                input_gradient = extracted_grads[gradient_position][0][0:src_lengths[0]-1] # first [] gets decoder/encoder grads, then batch (currently size 1), then we index into before the padding (though there shouldn't be any padding at the moment because batch size 1). The -1 is to ignore the padding
                candidate_input_tokens = hotflip_attack(input_gradient,
                                                          embedding_weight,
                                                          samples['net_input']['src_tokens'].cpu().numpy()[0],
                                                          num_candidates=200,
                                                          increase_loss=False)
            elif attack_mode == 'random':
                candidate_input_tokens = random_attack(embedding_weight,
                                                         samples['net_input']['src_tokens'].cpu().numpy()[0],
                                                         num_candidates=200)

            new_found_input_tokens = None
            batch_size = 64
            # take samples (of batch size 1) and repeat it batch_size times
            samples_repeated_by_batch = deepcopy(samples)
            samples_repeated_by_batch['ntokens'] *= batch_size
            samples_repeated_by_batch['target'] = samples_repeated_by_batch['target'].repeat(batch_size, 1)
            samples_repeated_by_batch['net_input']['prev_output_tokens'] = samples_repeated_by_batch['net_input']['prev_output_tokens'].repeat(batch_size, 1)
            samples_repeated_by_batch['net_input']['src_tokens'] = samples_repeated_by_batch['net_input']['src_tokens'].repeat(batch_size, 1)
            samples_repeated_by_batch['net_input']['src_lengths'] = samples_repeated_by_batch['net_input']['src_lengths'].repeat(batch_size, 1)

            all_inference_samples = [] # stores a list of batches of candidates
            current_batch_size = 0
            all_changed_positions = [] # stores all the changed_positions for each batch element
            current_batch_changed_position = []
            current_inference_samples = deepcopy(samples_repeated_by_batch) # stores one batch worth of candidates
            for index in range(len(candidate_input_tokens)): # for all the positions in the input
                for token_id in candidate_input_tokens[index]: # for all the candidates                    
                    if index == source_position: # don't edit the specific word in the source
                        continue
                    if changed_positions[index]: # if we have already changed this position, skip
                        continue
                    current_inference_samples['net_input']['src_tokens'][current_batch_size][index] = torch.LongTensor([token_id]).cuda().squeeze(0) # change on token                    
                    current_batch_size += 1
                    current_batch_changed_position.append(index) # save its changed position

                    if current_batch_size == batch_size: # batch is full
                        all_inference_samples.append(deepcopy(current_inference_samples))
                        current_inference_samples = deepcopy(samples_repeated_by_batch)
                        current_batch_size = 0
                        all_changed_positions.append(current_batch_changed_position)
                        current_batch_changed_position = []

            for inference_indx, inference_sample in enumerate(all_inference_samples):
                predictions = trainer.task.inference_step(generator, [trainer.get_model()],
                    inference_sample) # batched inference
                for prediction_indx, prediction in enumerate(predictions): # for all predictions
                    prediction = prediction[0]['tokens'].cpu()
                    # if prediction is the same, then save input
                    for idx, current_token in enumerate(prediction):
                        if current_token == desired_output_token: # found it!!!
                            print('found it!!!\n')
                            new_found_input_tokens = deepcopy(inference_sample['net_input']['src_tokens'][prediction_indx].unsqueeze(0))
                        #changed_positions[all_changed_positions[inference_indx][prediction_indx]] = True
                        #break # break twice
                #if new_found_input_tokens is not None:
                    #break

            # Update current input if the new caindidate flipped a position
            if new_found_input_tokens is not None:
                if attack_mode == 'random':
                    attack_mode = 'gradient'
                    #print('random worked, switching back to gradient')
                samples['net_input']['src_tokens'] = new_found_input_tokens

            # gradient is deterministic, so if it didnt flip another then its never going to
            else:
                if attack_mode == 'gradient': # once gradient fails, start using the decoder gradient
                    attack_mode = 'decoder_gradient'
                    #print('no more succesful flips, switching from gradient to decoder_gradient')
                elif attack_mode == 'decoder_gradient':
                    attack_mode = 'random'
                    break
                    #print('no more succesful flips, switching from decoder_gradient to random')
        num_total_samples += 1.0
        print(changed_positions)
        if any(changed_positions):
            num_samples_changed += 1.0
            num_tokens_changed += sum(changed_positions)
            total_num_tokens += len(changed_positions)
        print('\n')
    print('Total Num Samples', num_total_samples)
    print('Percent Samples Changed', num_samples_changed / num_total_samples)
    print('Percent Tokens Changed', num_tokens_changed / total_num_tokens)


#import nltk 
#from nltk.corpus import wordnet 
#antonyms = [] 
  
#for syn in wordnet.synsets("good"): 
#    for l in syn.lemmas(): 
#        if l.antonyms(): 
#            antonyms.append(l.antonyms()[0].name()) 
  
#print(set(antonyms))

parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
# make sure everything is reset before loading the model
args.reset_optimizer = True
args.reset_meters = True
args.reset_dataloader = True
args.reset_lr_scheduler = True
args.path = args.restore_file
main(args)

