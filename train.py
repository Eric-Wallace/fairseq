#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random
import numpy as np
from copy import deepcopy

import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter



extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

# returns the wordpiece embedding weight matrix
def get_embedding_weight(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            # TODO, this is hardcoded to be the size of the model's source embedding matrix
            if module.weight.shape[0] == 8848: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()

# add hooks for embeddings
def add_hooks(model):
    hook_registered = False
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):            
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

# def main(args, init_distributed=False):
#     utils.import_user_module(args)

#     assert args.max_tokens is not None or args.max_sentences is not None, \
#         'Must specify batch size either with --max-tokens or --max-sentences'

#     # Initialize CUDA and distributed training
#     if torch.cuda.is_available() and not args.cpu:
#         torch.cuda.set_device(args.device_id)
#     torch.manual_seed(args.seed)
#     if init_distributed:
#         args.distributed_rank = distributed_utils.distributed_init(args)

#     if distributed_utils.is_master(args):
#         checkpoint_utils.verify_checkpoint_directory(args.save_dir)

#     # Print args
#     print(args)

#     # Setup task, e.g., translation, language modeling, etc.
#     task = tasks.setup_task(args)

#     # Load valid dataset (we load training data below, based on the latest checkpoint)
#     for valid_sub_split in args.valid_subset.split(','):
#         task.load_dataset(valid_sub_split, combine=False, epoch=0)

#     # Build model and criterion
#     model = task.build_model(args)
#     criterion = task.build_criterion(args)

#     print(model)
#     print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
#     print('| num. model params: {} (num. trained: {})'.format(
#         sum(p.numel() for p in model.parameters()),
#         sum(p.numel() for p in model.parameters() if p.requires_grad),
#     ))

#     # Build trainer
#     trainer = Trainer(args, task, model, criterion)
#     print('| training on {} GPUs'.format(args.distributed_world_size))
#     print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
#         args.max_tokens,
#         args.max_sentences,
#     ))

#     # Load the latest checkpoint if one is available and restore the
#     # corresponding train iterator
#     extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

#     # Train until the learning rate gets too small
#     max_epoch = args.max_epoch or math.inf
#     max_update = args.max_update or math.inf
#     lr = trainer.get_lr()
#     train_meter = StopwatchMeter()
#     train_meter.start()
#     valid_losses = [None]
#     valid_subsets = args.valid_subset.split(',')
#     while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
#         # train for one epoch
#         train(args, trainer, task, epoch_itr)

#         if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
#             valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
#         else:
#             valid_losses = [None]

#         # only use first validation loss to update the learning rate
#         lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

#         # save checkpoint
#         if epoch_itr.epoch % args.save_interval == 0:
#             checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

#         if ':' in getattr(args, 'data', ''):
#             # sharded data: get train iterator for next epoch
#             epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
#     train_meter.stop()
#     print('| done training in {:.1f} seconds'.format(train_meter.sum))

def main(args, init_distributed=False):
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
   
    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)    

    # Evaluate without the trigger
    #print("Validation loss without trigger")
    #valid_subsets = args.valid_subset.split(',')
    #print(validate(args, trainer, task, epoch_itr, valid_subsets))

    # Initialize generator
    generator = task.build_generator(args)

    generate_trigger(args, trainer, epoch_itr)    
    
def generate_trigger(args, trainer, epoch_itr):    
    add_hooks(trainer.model) # add gradient hooks to embeddings    
    embedding_weight = get_embedding_weight(trainer.model) # save the embedding matrix

    # Initialize data iterator
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    

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
    for subset in subsets:       
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
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

def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()



