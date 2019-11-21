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
    new_token_ids = [[None]*num_candidates for _ in range(token_ids)]
    for token_id in range(len(token_ids)):
        for candidate_number in range(num_candidates):
            # rand token in the embedding matrix
            rand_token = np.random.randint(embedding_matrix.shape[0])
            new_token_ids[token_id][candidate_number] = rand_token
    return new_token_ids

# runs the samples through the model and fills extracted_grads with the gradient w.r.t. the embedding
def get_input_grad(trainer, samples):
    trainer._set_seed()
    trainer.get_model().eval() # we want grads from eval() model, to turn off dropout and stuff
    trainer.criterion.train()
    trainer.zero_grad()

    # fills extracted_grads with the gradient w.r.t. the embedding
    sample = trainer._prepare_sample(samples)
    loss, _, __, prediction = trainer.criterion(trainer.get_model(), sample, return_prediction=True)
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
    flip_tokens(args, trainer, generator)

def flip_tokens(args, trainer, generator):
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

    attack_mode = 'gradient' # gradient or random flipping
    for i, samples in enumerate(itr): # for the whole validation set (could be fake data if its interactive model)
        if args.interactive_attacks: # get user input and build samples            
            user_input = input('Enter your sentence:\n')
            # tokenize input and get lengths
            tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(user_input)).long().unsqueeze(dim=0)
            length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])])            
            # build samples and set their targets with the model predictions
            samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])}
        
        changed_positions = [False] * samples['ntokens'] # if a position is already changed, don't change it again. ntokens = size for batch size 1
        samples, original_prediction = run_inference_and_overwrite_samples(samples)        
        for i in range(samples['ntokens'] * 3): # this many iters over a single batch. Gradient attack will early stop
            print('Current Model Input: ', bpe.decode(trainer.task.source_dictionary.string(samples['net_input']['src_tokens'].cpu()[0], None)))
            samples, predictions = run_inference_and_overwrite_samples(samples)
            print('Current Model Predict:', bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(predictions), None)))
            src_lengths, _ = get_input_grad(trainer, samples) # gradient is now filled
            if attack_mode == 'gradient':
                input_gradient = extracted_grads[0][0][0:src_lengths[0]] # first [0] removes list from hook, then batch, then indexes into before the padding (though there shouldn't be any padding at the moment)
                candidate_input_tokens = hotflip_attack(input_gradient,
                                                          embedding_weight,
                                                          samples['net_input']['src_tokens'].cpu().numpy()[0],
                                                          num_candidates=30,
                                                          increase_loss=False)
            elif attack_mode == 'random':
                candidate_input_tokens = random_attack(embedding_weight,
                                                         samples['net_input']['src_tokens'].cpu().numpy()[0],
                                                         num_candidates=20)
            #curr_best_loss = -999999999
            new_found_input_tokens = None
            #curr_best_input_tokens_position_changed = None
            for index in range(len(candidate_input_tokens)): # for all the positions in the input
                for token_id in candidate_input_tokens[index]: # for all the candidates
                    if changed_positions[index] or new_found_input_tokens is not None: # if we have already changed this position, skip
                        continue

                    # replace one token with new candidate
                    original_src = deepcopy(samples['net_input']['src_tokens'])                    
                    samples['net_input']['src_tokens'][0][index] = torch.LongTensor([token_id]).cuda().squeeze(0)

                    predictions = trainer.task.inference_step(generator, [trainer.get_model()], samples)[0][0]['tokens'].cpu()                    
                    # if prediction is the same, then save input
                    if predictions.shape == original_prediction.shape and all(torch.eq(predictions,original_prediction)):
                        # if someone has already succesfully flipped a position, then only update if the new one is lower loss
                        #if curr_best_input_tokens_position_changed is not None:# and curr_loss > curr_best_loss:
                             #continue
                        #curr_best_loss = curr_loss
                        new_found_input_tokens = deepcopy(samples['net_input']['src_tokens'])
                        changed_positions[index] = True                        
                        #curr_best_input_tokens_position_changed = index                        
                    # restore input
                    samples['net_input']['src_tokens'] = original_src

            # Update current input if the new candidate flipped a position
            #if curr_best_input_tokens_position_changed is not None:
                #input_token_ids = deepcopy(curr_best_input_tokens)
                #changed_positions[curr_best_input_tokens_position_changed] = True
            if new_found_input_tokens is not None:                
                samples['net_input']['src_tokens'] = new_found_input_tokens                

            # gradient is deterministic, so if it didnt flip another then its never going to
            elif attack_mode == 'gradient':
                print('no more succesful flips\n')
                break

parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
# make sure everything is reset before loading the model
args.reset_optimizer = True
args.reset_meters = True
args.reset_dataloader = True
args.reset_lr_scheduler = True
args.path = args.restore_file
main(args)
