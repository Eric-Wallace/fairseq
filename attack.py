import numpy as np
from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer
from nltk.corpus import wordnet

extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

# returns the encoder embedding weight matrix
def get_embedding_weight(model, bpe_vocab_size):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == bpe_vocab_size:
                return module.weight.detach()
    exit("Embedding matrix not found")

# add a hook to the encoder embedding matrix
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

# runs the samples through the model. If no_backwards is False, calls backwards() to fill extracted_grads
# with the gradient w.r.t. the embedding. If reduce_loss is False, does not reduce the loss across the batch
# dimension. 
def get_input_grad_predict_loss(trainer, samples, mask=None, no_backwards=False, reduce_loss=True):
    trainer._set_seed()
    trainer.get_model().eval() # we want grads from eval() to turn off dropout and stuff
    trainer.criterion.train() # TODO, do I want train() here? It might not need to be set
    trainer.zero_grad()

    # fills extracted_grads with the gradient w.r.t. the embedding
    sample = trainer._prepare_sample(samples)
    loss, _, __, prediction = trainer.criterion(trainer.get_model(), sample, return_prediction=True, mask=mask, reduce=reduce_loss)
    if not no_backwards:
        trainer.optimizer.backward(loss)
    return sample['net_input']['src_lengths'], prediction.max(2)[1].squeeze().detach().cpu(), loss.detach().cpu()

# run model to get predictions, If no_overwrite is False, the targets of samples are overwritten with the predictions
def run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=False):
    # move samples to GPU
    if torch.cuda.is_available() and not args.cpu:
        samples['net_input']['src_tokens'] = samples['net_input']['src_tokens'].cuda()
        samples['net_input']['src_lengths'] = samples['net_input']['src_lengths'].cuda()
        if 'target' in samples:
            samples['target'] = samples['target'].cuda()
            samples['net_input']['prev_output_tokens'] = samples['net_input']['prev_output_tokens'].cuda()

    # run model
    translations = trainer.task.inference_step(generator, [trainer.get_model()], samples)
    predictions = translations[0][0]['tokens'].cpu()
    if no_overwrite: # don't overwrite samples and just return
        return samples, predictions

    samples['target'] = translations[0][0]['tokens'].unsqueeze(dim=0) # overwrite samples
    # prev_output_tokens is the right rotated version of the target
    samples['net_input']['prev_output_tokens'] = torch.cat((samples['target'][0][-1:], 
                                                            samples['target'][0][:-1]), dim=0).unsqueeze(dim=0)

    return samples, predictions

# take samples (of batch size 1) and repeat it batch_size times to do batched inference / loss calculation
def build_inference_samples(samples, batch_size, args, candidate_input_tokens, changed_positions, trainer, bpe, num_trigger_tokens=None):    
    # copy and repeat the samples batch size times
    samples_repeated_by_batch = deepcopy(samples)
    samples_repeated_by_batch['ntokens'] *= batch_size
    samples_repeated_by_batch['target'] = samples_repeated_by_batch['target'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['prev_output_tokens'] = samples_repeated_by_batch['net_input']['prev_output_tokens'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['src_tokens'] = samples_repeated_by_batch['net_input']['src_tokens'].repeat(batch_size, 1)
    samples_repeated_by_batch['net_input']['src_lengths'] = samples_repeated_by_batch['net_input']['src_lengths'].repeat(batch_size, 1)
    samples_repeated_by_batch['nsentences'] = batch_size

    all_inference_batches = [] # stores a list of batches of candidates
    all_changed_positions_batches = [] # stores all the changed_positions for each batch element
    
    current_batch_size = 0
    current_batch_changed_positions = []
    current_inference_batch = deepcopy(samples_repeated_by_batch) # stores one batch worth of candidates
    for index in range(len(candidate_input_tokens)): # for all the positions in the input
        for token_id in candidate_input_tokens[index]: # for all the candidates                            
            if changed_positions[index]: # if we have already changed this position, skip.
                continue
            
            current_inference_batch['net_input']['src_tokens'][current_batch_size][index_to_use] = torch.LongTensor([token_id]).squeeze(0) # change one token           
            current_batch_size += 1
            current_batch_changed_positions.append(index_to_use) # save its changed position

            if current_batch_size == batch_size: # batch is full
                all_inference_batches.append(deepcopy(current_inference_batch))
                current_inference_batch = deepcopy(samples_repeated_by_batch)
                current_batch_size = 0
                all_changed_positions_batches.append(current_batch_changed_positions)
                current_batch_changed_positions = []
    
    return all_inference_batches, all_changed_positions_batches

# uses a gradient-based attack to get the candidates for each position in the input
def get_attack_candidates(trainer, samples, embedding_weight, mask=None, increase_loss=False, num_candidates=400):
    src_lengths, _, __ = get_input_grad_predict_loss(trainer, samples, mask=mask) # gradient is now filled
    
    # for models with shared embeddings, position 1 in extracted_grads will be the encoder grads, 0 is decoder
    if len(extracted_grads) > 1:
        gradient_position = 1
    else:
        gradient_position = 0
    
    # first [] gets decoder/encoder grads, then gets ride of batch.
    # then we index into before the padding (though there shouldn't be any padding at the moment because batch size 1).
    # The -1 is to ignore the pad symbol.
    input_gradient = extracted_grads[gradient_position][0][0:src_lengths[0]-1].cpu()     
    embedding_matrix = embedding_matrix.cpu()    
    input_gradient = input_gradient.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", (input_gradient, embedding_matrix))
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1 # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    else:
        _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
        return best_at_each_step[0].detach().cpu().numpy()


def get_user_input(trainer, bpe):
    user_input = input('Enter your sentence: ')
    
    # tokenize input and get lengths
    tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(user_input)).long().unsqueeze(dim=0)
    length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])])
    
    # build samples and set their targets to be the model predictions
    samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])}
    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    
    # check if the user input a token with is an UNK
    for token in tokenized_bpe_input[0]:
        if torch.eq(token, bpe_vocab_size) or torch.gt(token, bpe_vocab_size):
            return 'invalid' 
    return samples


def main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    
    # make sure everything is reset before loading the model
    args.reset_optimizer = True; args.reset_meters = True; args.reset_dataloader = True; args.reset_lr_scheduler = True;
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.path = args.restore_file
    args.max_sentences_valid = 1  # We attack batch size 1 at the moment
    args.beam = 1 # beam size 1 for inference on the model
    utils.import_user_module(args)

    # setup task, model, loss function, and trainer
    task = tasks.setup_task(args)
    if not args.interactive_attacks:
        for valid_sub_split in args.valid_subset.split(','): # load validation data
            task.load_dataset(valid_sub_split, combine=False, epoch=0)
    assert len(args.path.split(':')) == 1 # TODO, only handles one model at the moment
    models, _= checkpoint_utils.load_model_ensemble(args.path.split(':'), arg_overrides={}, task=task)
    model = models[0] # only one model

    if torch.cuda.is_available() and not args.cpu:
        assert torch.cuda.device_count() == 1 # only works on 1 GPU for now
        torch.cuda.set_device(0)
        model.cuda()    
    model.make_generation_fast_(beamable_mm_beam_size=args.beam, need_attn=False)

    criterion = task.build_criterion(args)
    trainer = Trainer(args, task, model, criterion)
    generator = task.build_generator(args)

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
    
    num_samples_changed = 0.0
    num_total_samples = 0.0
    num_tokens_changed = 0.0
    total_num_tokens = 0.0
    for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
        changed_positions = malicious_nonsense(samples, args, trainer, generator, embedding_weight, itr, bpe, i)        
        if changed_positions is None: # error, skip sample            
            continue

        num_total_samples += 1.0  
        if any(changed_positions):
            num_samples_changed += 1.0
            num_tokens_changed += sum(changed_positions)
            total_num_tokens += len(changed_positions)
        
    if num_total_samples > 0.0:         
         print('Total Num Samples', num_total_samples)
         print('Percent Samples Changed', num_samples_changed / num_total_samples)
         print('Percent Tokens Changed', num_tokens_changed / total_num_tokens)


def malicious_nonsense(samples, args, trainer, generator, embedding_weight, itr, bpe, i):
    if args.interactive_attacks: # get user input and build samples
        samples = get_user_input(trainer, bpe)    
    if samples == 'invalid': # user input an UNK token
        return None

    original_samples = deepcopy(samples)    
    samples, original_prediction = run_inference_and_maybe_overwrite_samples(trainer, generator, samples)    

    # if a position is already changed, don't change it again.
    # [False] for the sequence length, but minus -1 to ignore pad
    changed_positions = [False] * (samples['net_input']['src_tokens'].shape[1] - 1) 
    while True: # we break when we can't find another good token        
        assert samples['net_input']['src_tokens'].cpu().numpy()[0][-1] == 2 # make sure pad it always there                
        print('Current Tokens', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
        samples, predictions = run_inference_and_maybe_overwrite_samples(trainer, generator, samples, no_overwrite=True)
        print('Current Prediction' bpe.decode(trainer.task.target_dictionary.string(torch.LongTensor(predictions), None)))

        # clear grads, compute new grads, and get candidate tokens
        global extracted_grads
        extracted_grads = []
        candidate_input_tokens = get_attack_candidates(trainer, samples, embedding_weight, mask=None)

        # batch up all the candidate input tokens
        new_found_input_tokens = None
        batch_size = 64
        all_inference_batches, all_changed_positions = build_inference_samples(samples, batch_size, args, candidate_input_tokens, changed_positions, trainer, bpe)

        # for all the possible new samples
        for inference_indx, inference_batch in enumerate(all_inference_batches):
            # run the model on a batch of samples
            predictions = trainer.task.inference_step(generator, [trainer.get_model()],
                inference_batch)
            for prediction_indx, prediction in enumerate(predictions): # for all predictions in that batch
                prediction = prediction[0]['tokens'].cpu()
                # if prediction is the same, then save input
                if prediction.shape == original_prediction.shape and all(torch.eq(prediction,original_prediction)):
                    # if the "new" candidate is actually the same as the current tokens, skip it
                    if all(torch.eq(inference_batch['net_input']['src_tokens'][prediction_indx],samples['net_input']['src_tokens'].squeeze(0))):
                        continue
                    new_found_input_tokens = deepcopy(inference_batch['net_input']['src_tokens'][prediction_indx].unsqueeze(0))
                    changed_positions[all_changed_positions[inference_indx][prediction_indx]] = True
                    break # break twice
            if new_found_input_tokens is not None:
                break

        # Update current input if the new candidate flipped a position
        if new_found_input_tokens is not None:                            
            samples['net_input']['src_tokens'] = new_found_input_tokens
        
        # gradient is deterministic, so if it didnt flip another then its never going to
        else: 
            break
        
    return changed_positions

if __name__ == '__main__':
    main()