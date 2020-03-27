from copy import deepcopy
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer

# returns the encoder embedding weight matrix
def get_embedding_weight(model, bpe_vocab_size):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == bpe_vocab_size:
                return module.weight.detach().cpu()
    exit("Embedding matrix not found")

# extracted_grads will store the gradients for the input embeddings, and is filled by the hook below.
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

# add a hook to the encoder embedding matrix. Used for getting the gradient w.r.t. the embeddings
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

def get_user_input(trainer, bpe):
    user_input = input('Enter the input sentence that you want to turn into nonsense: ')

    # tokenize input and get lengths
    tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(user_input)).long().unsqueeze(dim=0)

    # check if the user input a token with is an UNK
    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    for token in tokenized_bpe_input[0]:
        if torch.eq(token, bpe_vocab_size) or torch.gt(token, bpe_vocab_size): # >= max vocab size
            print('You entered an UNK token for your model, please try again. This usually occurs when (1) you entered '
                ' unicode or other strange symbols, (2) your model uses a lowercased dataset but you entered uppercase, or '
                ' (3) your model is expecting apostrophies as &apos; and quotes as &quot;')
            return None

    length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])])
    # build samples which is input to the model
    samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])}

    return samples


# runs the samples through the model, computes the cross entropy loss, and then backprops.
# This fills the extracted_grads list with the gradient w.r.t. the input embeddings.
def get_input_grad(trainer, samples):
    trainer._set_seed()
    trainer.get_model().eval() # we want grads from eval() to turn off dropout and stuff
    trainer.zero_grad()

    sample = trainer._prepare_sample(samples)
    loss, _, _, = trainer.criterion(trainer.get_model(), sample)
    trainer.optimizer.backward(loss) # fills extracted_grads with the gradient w.r.t. the embedding
    return sample['net_input']['src_lengths']


# take samples (which is batch size 1) and repeat it batch_size times to do batched inference / loss calculation
# for all of the possible attack candidates
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

            original_token = deepcopy(current_inference_batch['net_input']['src_tokens'][current_batch_size][index]) # save the original token, might be used below if there is an error
            current_inference_batch['net_input']['src_tokens'][current_batch_size][index] = torch.LongTensor([token_id]).squeeze(0) # change one token

            # there are cases where making a BPE swap would cause the BPE segmentation to change.
            # in other words, the input we are using would be invalid because we are using an old segmentation
            # for these cases, we just skip those candidates
            string_input_tokens = bpe.decode(trainer.task.source_dictionary.string(current_inference_batch['net_input']['src_tokens'][current_batch_size].cpu(), None))
            retokenized_string_input_tokens = trainer.task.source_dictionary.encode_line(bpe.encode(string_input_tokens)).long().unsqueeze(dim=0)
            if torch.cuda.is_available() and not trainer.args.cpu:
                retokenized_string_input_tokens = retokenized_string_input_tokens.cuda()
            if len(retokenized_string_input_tokens[0]) != len(current_inference_batch['net_input']['src_tokens'][current_batch_size]) or \
                not torch.all(torch.eq(retokenized_string_input_tokens[0], current_inference_batch['net_input']['src_tokens'][current_batch_size])):
                    # undo the token we replaced and move to the next candidate
                    current_inference_batch['net_input']['src_tokens'][current_batch_size][index] = original_token
                    continue

            current_batch_size += 1
            current_batch_changed_positions.append(index) # save its changed position

            if current_batch_size == batch_size: # batch is full
                all_inference_batches.append(deepcopy(current_inference_batch))
                current_inference_batch = deepcopy(samples_repeated_by_batch)
                current_batch_size = 0
                all_changed_positions_batches.append(current_batch_changed_positions)
                current_batch_changed_positions = []

    return all_inference_batches, all_changed_positions_batches

# uses a gradient-based attack to get the candidates for each position in the input
def get_attack_candidates(trainer, samples, embedding_weight, num_gradient_candidates=500):
    # clear grads, compute new grads
    global extracted_grads; extracted_grads = []
    src_lengths = get_input_grad(trainer, samples) # input embedding gradient is now filled

    # for models with shared embeddings, position 1 in extracted_grads will be the encoder grads, 0 is decoder
    if len(extracted_grads) > 1:
        gradient_position = 1
    else:
        gradient_position = 0

    # first [] gets decoder/encoder grads, then gets ride of batch (we have batch size 1)
    # then we index into before the padding (though there shouldn't be any padding because we do batch size 1).
    # then the -1 is to ignore the pad symbol.
    assert len(extracted_grads) <= 2 and len(extracted_grads[gradient_position]) == 1 # make sure gradients are not accumulating
    input_gradient = extracted_grads[gradient_position][0][0:src_lengths[0]-1].cpu()
    input_gradient = input_gradient.unsqueeze(0)
    # the first-order taylor expansion (i.e., we are just dot producting the gradient with the embedding)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", (input_gradient, embedding_weight))
    gradient_dot_embedding_matrix *= -1 # flip the gradient around because we want to decrease the loss

    if num_gradient_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_gradient_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    else:
        _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
        return best_at_each_step[0].detach().cpu().numpy()