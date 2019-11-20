#!/usr/bin/env python3 -u

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
def get_embedding_weight(model):
    for module in model.modules():
        # TODO, hardcoded number here for the encoder weight matrix
        #
        #
        #
        if isinstance(module, torch.nn.Embedding):            
            if module.weight.shape[0] == 8848: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()

# add hooks for embeddings
def add_hooks(model):
    hook_registered = False
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):            
            # TODO, same thing here
            if module.weight.shape[0] == 8848: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)
                hook_registered = True
    if not hook_registered:
        exit("Embedding matrix not found")


def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py
    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.
    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids),
                                                         embedding_matrix).detach().unsqueeze(0)
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))
    gradient_dot_trigger_embeds = torch.einsum("bij,bij->bi",
                                               (averaged_grad, trigger_token_embeds)).unsqueeze(-1)
    final_taylor_approximation = gradient_dot_trigger_embeds - gradient_dot_embedding_matrix
    if increase_loss:
        final_taylor_approximation *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(final_taylor_approximation, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = final_taylor_approximation.max(2)
    return best_at_each_step[0].detach().cpu().numpy()


def random_attack(embedding_matrix, trigger_token_ids, num_candidates=1):
    """
    Randomly search over the vocabulary. Gets num_candidates random samples and returns all of them.
    """
    embedding_matrix = embedding_matrix.cpu()
    new_trigger_token_ids = [[None]*num_candidates for _ in range(len(trigger_token_ids))]
    for trigger_token_id in range(len(trigger_token_ids)):
        for candidate_number in range(num_candidates):
            # rand token in the embedding matrix
            rand_token = np.random.randint(embedding_matrix.shape[0])
            new_trigger_token_ids[trigger_token_id][candidate_number] = rand_token
    return new_trigger_token_ids

def main(args):
    utils.import_user_module(args)

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(criterion)    
   
    trainer = Trainer(args, task, model, criterion)
   
    #_, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)    

    # Evaluate without the trigger
    #print("Validation loss without trigger")
    #valid_subsets = args.valid_subset.split(',')
    #print(validate(args, trainer, task, epoch_itr, valid_subsets))

    generate_trigger(args, trainer, args.valid_subset.split(','))    
    
def generate_trigger(args, trainer, valid_subsets):    
    add_hooks(trainer.model) # add gradient hooks to embeddings    
    embedding_weight = get_embedding_weight(trainer.model) # save the embedding matrix

    assert len(valid_subsets) == 0
    itr = task.get_batch_iterator(dataset=task.dataset(subset),
              max_tokens=args.max_tokens_valid,
              max_sentences=args.max_sentences_valid,
              max_positions=utils.resolve_max_positions(
                  task.max_positions(),
                  trainer.get_model().max_positions(),
              ),
              ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
              required_batch_size_multiple=args.required_batch_size_multiple,
              seed=args.seed,
              num_shards=args.distributed_world_size,
              shard_id=args.distributed_rank,
              num_workers=args.num_workers,
          ).next_epoch_itr(shuffle=False)
    
    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # initialize trigger
    num_trigger_tokens = 10 
    trigger_token_ids = np.random.randint(8848, size=num_trigger_tokens)    
    best_loss = -1
    attack_mode = 'random'
    for i, samples in enumerate(itr):                    
        # this many iters over a single batch
        for i in range(100):
            if attack_mode == 'gradient':
                # get gradient
                src_lengths = trainer.get_trigger_grad(samples, trigger_token_ids)            
                # sum gradient across the different scattered areas based on the src length
                averaged_grad = None            
                for indx, grad in enumerate(extracted_grads[0]):
                    grad_for_trigger = grad[src_lengths[indx]: src_lengths[indx] + num_trigger_tokens]                 
                    if indx == 0:
                        averaged_grad = grad_for_trigger
                    else:
                        averaged_grad += grad_for_trigger                        
                # get the top candidates using first-order approximation
                candidate_trigger_tokens = hotflip_attack(averaged_grad,
                                                          embedding_weight,
                                                          trigger_token_ids,
                                                          num_candidates=20)  
            elif attack_mode == 'random':
                candidate_trigger_tokens = random_attack(embedding_weight,
                                                         trigger_token_ids,
                                                         num_candidates=20)
            else:
                exit("attack_mode")
            curr_best_loss = -1
            curr_best_trigger_tokens = None            
            for index in range(len(candidate_trigger_tokens)):
                for token_id in candidate_trigger_tokens[index]:
                    # replace one token with new candidate
                    temp_candidate_trigger_tokens = deepcopy(trigger_token_ids)
                    temp_candidate_trigger_tokens[index] = token_id

                    # get loss, update current best if its lower loss                                        
                    curr_loss = trainer.get_trigger_loss(samples, temp_candidate_trigger_tokens).detach().cpu()
                    if curr_loss > curr_best_loss:
                        curr_best_loss = curr_loss
                        curr_best_trigger_tokens = deepcopy(temp_candidate_trigger_tokens)

            # Update overall best if the best current candidate is better
            if curr_best_loss > best_loss:                
                best_loss = curr_best_loss
                trigger_token_ids = deepcopy(curr_best_trigger_tokens)
                print("Training Loss: " + str(best_loss.data.item()))                
                print(decode_fn(trainer.task.source_dictionary.string(torch.LongTensor(trigger_token_ids), None)))
        validate_trigger(args, trainer, trainer.task, trigger_token_ids)

def validate_trigger(args, trainer, task, trigger):    
    subsets = args.valid_subset.split(',')
    total_loss = None
    total_samples = 0.0
    assert len(subsets) == 1
    itr = task.get_batch_iterator(dataset=task.dataset(subset),
        max_tokens=args.max_tokens_valid,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    for i, samples in enumerate(itr):
        loss = trainer.get_trigger_loss([samples], trigger).detach().cpu()
        if total_loss is not None:
            total_loss += loss 
        else:
            total_loss = loss
        total_samples += 1.0
        
    print("Validation Loss Using First Token Cross Entropy: ", total_loss / total_samples)

parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
args.reset_optimizer=True
main(args)

#python dont_flip.py data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --share-decoder-input-output-embed --criterion first_token_cross_entropy --max-tokens 250 --restore-file checkpoints/checkpoint_best.pt --reset-optimizer
