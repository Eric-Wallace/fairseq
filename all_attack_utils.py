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
def get_input_grad(trainer, samples, target_mask=None, no_backwards=False, reduce_loss=True):
    trainer._set_seed()
    trainer.get_model().eval() # we want grads from eval() model, to turn off dropout and stuff
    trainer.criterion.train()
    trainer.zero_grad()

    # fills extracted_grads with the gradient w.r.t. the embedding
    sample = trainer._prepare_sample(samples)
    loss, _, __, prediction = trainer.criterion(trainer.get_model(), sample, return_prediction=True, target_mask=target_mask, reduce=reduce_loss)
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

            original_token = deepcopy(current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use])
            current_inference_samples['net_input']['src_tokens'][current_batch_size][index_to_use] = torch.LongTensor([token_id]).squeeze(0) # change onetoken

            # check if the BPE has changed, and if so, replace the samples            
            string_input_tokens = bpe.decode(trainer.task.source_dictionary.string(current_inference_samples['net_input']['src_tokens'][current_batch_size].cpu(), None))
            retokenized_string_input_tokens = trainer.task.source_dictionary.encode_line(bpe.encode(string_input_tokens)).long().unsqueeze(dim=0)
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

def get_attack_candidates(trainer, samples, attack_mode, embedding_weight, target_mask=None, increase_loss=False):
    # clear grads, compute new grads, and get candidate tokens
    global extracted_grads
    extracted_grads = [] # clear old extracted_grads
    src_lengths, _, __ = get_input_grad(trainer, samples, target_mask=target_mask) # gradient is now filled
    if 'gradient' in attack_mode:
        if len(extracted_grads) > 1: # this is for models with shared embedding
            # position [1] in extracted_grads is the encoder embedding grads, [0] is decoder
            if attack_mode == 'gradient':
                gradient_position = 1
            elif attack_mode == 'decoder_gradient':
                gradient_position = 0
        else:
            gradient_position = 0
        assert len(extracted_grads) <= 2 and len(extracted_grads[gradient_position]) == 1 # make sure gradients are not accumulating
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


def update_attack_mode_state_machine(attack_mode):
    if attack_mode == 'gradient': # once gradient fails, start using the decoder gradient
        attack_mode = 'decoder_gradient'        
    elif attack_mode == 'decoder_gradient':
        attack_mode = 'random'        
    return attack_mode
