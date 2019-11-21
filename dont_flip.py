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


def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1):
    """    
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
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik", (averaged_grad, embedding_matrix))
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()

# random search, returns num_candidates random samples.
def random_attack(embedding_matrix, trigger_token_ids, num_candidates=1):    
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
    # args.max_positions = TODO
    # args.criterion=''
    args.share_all_embeddings = True # TODO, make this model-dependent    
    args.max_tokens_valid = 256 # TODO, for now we do batch size with small sentences
    args.max_positions = 256
    args.skip_invalid_size_inputs_valid_test = True
    args.max_sentences_valid = 1   
    args.update_freq=1

    # Initialize CUDA
    if torch.cuda.is_available() and not args.cpu:
        assert torch.cuda.device_count() == 1 # only works on 1 GPU for now
        #torch.cuda.set_device(args.device_id)
        torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
        
    # setup task, model, loss function, and trainer
    task = tasks.setup_task(args)
    for valid_sub_split in args.valid_subset.split(','): # load validation data
        task.load_dataset(valid_sub_split, combine=False, epoch=0)
    #model = task.build_model(args)
    args.model_overrides='{}'
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    ) 
    model = models[0]
    criterion = task.build_criterion(args)       
    trainer = Trainer(args, task, model, criterion)   
    print(args); print(task); print(model); print(criterion)

    generate_trigger(args, trainer, args.valid_subset.split(','))    
    
def generate_trigger(args, trainer, valid_subsets):    
    bpe_vocab_size = trainer.get_model().encoder.embed_tokens.weight.shape[0]
    add_hooks(trainer.model, bpe_vocab_size) # add gradient hooks to embeddings    
    embedding_weight = get_embedding_weight(trainer.model, bpe_vocab_size) # save the embedding matrix

    assert len(valid_subsets) == 1 # only one validation subset handled
    subset = valid_subsets[0]
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
    
    # Handle tokenization and BPE
    #tokenizer = encoders.build_tokenizer(args)
    #bpe = encoders.build_bpe(args)

    #def encode_fn(x):
    #    if tokenizer is not None:
    #        x = tokenizer.encode(x)
    #    if bpe is not None:
    #        x = bpe.encode(x)
    #    return x

    #def decode_fn(x):
    #    if bpe is not None:
    #        x = bpe.decode(x)
    #    if tokenizer is not None:
    #        x = tokenizer.decode(x)
    #    return x

    # initialize trigger
    #num_trigger_tokens = 10
    #trigger_token_ids = np.random.randint(bpe_vocab_size, size=num_trigger_tokens)    
    best_loss = -99999999999
    attack_mode = 'gradient'
    for i, samples in enumerate(itr): # for the whole dataset
        # original ids are the original sample ids
        input_token_ids = samples['net_input']['src_tokens'].cpu().numpy()
        input_token_ids = input_token_ids[0] # batch size 1 
        if len(input_token_ids) <15:
           continue
        changed_positions = [False] * len(input_token_ids) 
        
        # get original prediction
        _src_lengths, predictions = trainer.get_trigger_grad(samples, None)#input_token_ids)
        del _src_lengths 
        predictions = predictions.detach().cpu()
        original_prediction = predictions.max(2)[1].squeeze()
        print('\n\n')
        print(original_prediction)

        for i in range(100): # this many iters over a single batch
            print('input ids', input_token_ids)
            print('positions', changed_positions)
            if attack_mode == 'gradient':                 
                src_lengths, predictions = trainer.get_trigger_grad(samples, None)#input_token_ids) # get gradient
                print('predictions', predictions.max(2)[1].squeeze())
                del predictions
                # sum gradient across the different scattered areas based on the src length
                averaged_grad = None            
                assert len(extracted_grads[0]) == 1 # batch size 1 fornow
                for indx, grad in enumerate(extracted_grads[0]):
                    #grad_for_trigger = grad[src_lengths[indx]: src_lengths[indx] + num_trigger_tokens]
                    grad_for_trigger = grad[0:src_lengths[indx]]
                    if averaged_grad is None:
                        averaged_grad = grad_for_trigger
                    else:
                        averaged_grad += grad_for_trigger                        
                # get the top candidates using first-order approximation
                candidate_trigger_tokens = hotflip_attack(averaged_grad,
                                                          embedding_weight,
                                                          input_token_ids,
                                                          num_candidates=10,
                                                          increase_loss=True)
            elif attack_mode == 'random':
                candidate_trigger_tokens = random_attack(embedding_weight,
                                                         input_token_ids,
                                                         num_candidates=20)
            else:
                exit("attack_mode")
            curr_best_loss = -999999999
            curr_best_input_tokens = None
            curr_best_trigger_tokens_position_changed = None
            for index in range(len(candidate_trigger_tokens)):
                for token_id in candidate_trigger_tokens[index]:
                    if changed_positions[index]:
                        continue
                    # replace one token with new candidate
                    temp_candidate_trigger_tokens = deepcopy(input_token_ids)
                    temp_candidate_trigger_tokens[index] = token_id

                    # get loss, update current best if its lower loss                                        
                    curr_loss, predictions = trainer.get_trigger_loss(samples, temp_candidate_trigger_tokens)
                    curr_loss = curr_loss.detach().cpu()
                    predictions = predictions.detach().cpu().max(2)[1].squeeze() # 2 is the logit dimension, [1] is the indices of the max
                    if all(torch.eq(predictions,original_prediction)):# and curr_loss < curr_best_loss: # less loss because decrease
                        if curr_best_trigger_tokens_position_changed is not None and curr_loss > curr_best_loss:
                             continue # for tie breaking, take the one with lower loss
                        curr_best_loss = curr_loss
                        curr_best_trigger_tokens = deepcopy(temp_candidate_trigger_tokens)
                        curr_best_trigger_tokens_position_changed = index 
          
            # Update overall best if the best current candidate is better
            if curr_best_trigger_tokens_position_changed is not None:#curr_best_loss < best_loss: # less than               
                #best_loss = curr_best_loss
                input_token_ids = deepcopy(curr_best_trigger_tokens)
                changed_positions[curr_best_trigger_tokens_position_changed] = True
                #print('new best loss', best_loss)
                #print("Training Loss: " + str(best_loss.data.item()))                
                #print(decode_fn(trainer.task.source_dictionary.string(torch.LongTensor(trigger_token_ids), None)))
            else:
                if attack_mode == 'gradient':
                    print('breaking\n')
                    break # gradient is deterministic, so if it didnt flip another then its never going to
        #validate_trigger(args, trainer, trainer.task, trigger_token_ids)

def validate_trigger(args, trainer, task, trigger):    
    subsets = args.valid_subset.split(',')
    total_loss = None
    total_samples = 0.0
    assert len(subsets) == 1
    subset = subsets[0]
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
        
    print("Validation Loss Using Trigger: ", total_loss / total_samples)
#parser = options.get_generation_parser(interactive=False)
parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
args.reset_optimizer = True # make sure everything is reset before loading the model
args.reset_meters = True
args.reset_dataloader = True
args.reset_lr_scheduler = True
args.path = args.restore_file
main(args)

# python dont_flip.py google/ --arch transformer_vaswani_wmt_en_de_big --restore-file checkpoint_best.pt
