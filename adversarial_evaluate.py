import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import iterators, encoders
from fairseq.trainer import Trainer

def read_file_input(trainer, bpe):
    user_input = input('')
    if user_input is None or user_input == ' ' or user_input.strip() == "" or user_input == '\n':
        return None
    # tokenize input and get lengths
    tokenized_bpe_input = trainer.task.source_dictionary.encode_line(bpe.encode(user_input)).long().unsqueeze(dim=0).cuda()
    length_user_input = torch.LongTensor([len(tokenized_bpe_input[0])]).cuda()
    # build samples and set their targets with the model predictions
    samples = {'net_input': {'src_tokens': tokenized_bpe_input, 'src_lengths': length_user_input}, 'ntokens': len(tokenized_bpe_input[0])}
    return samples

def main(args):
    utils.import_user_module(args)
    args.max_sentences_valid = 1  # batch size 1 at the moment

    # Initialize CUDA    
    torch.manual_seed(args.seed)

    # setup task, model, loss function, and trainer
    task = tasks.setup_task(args)    
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

    # Handle BPE
    bpe = encoders.build_bpe(args)
    assert bpe is not None
    attack(trainer, generator, bpe)

def attack(trainer, generator, bpe):
    num_prediction_matched = 0.0
    num_total = 0.0
    while True:
        prediction_matched = malicious_nonsense(trainer, generator, bpe)
        if prediction_matched == None:
            break
        elif prediction_matched:
            num_prediction_matched += 1
        num_total += 1
    
    print('Total Inputs', num_total)
    print('Num Matched', num_prediction_matched)

def malicious_nonsense(trainer, generator, bpe):
    orig_sample = read_file_input(trainer, bpe)
    if orig_sample is None:
        return None
    adv_sample = read_file_input(trainer, bpe)
    if adv_sample is None:
        return None
    
    print('Orig Input ', bpe.decode(trainer.task.source_dictionary.string(orig_sample['net_input']['src_tokens'].cpu()[0], None)))
    print('Adv Input  ', bpe.decode(trainer.task.source_dictionary.string(adv_sample['net_input']['src_tokens'].cpu()[0], None)))
    orig_prediction = trainer.task.inference_step(generator, [trainer.get_model()], orig_sample)    
    adv_prediction = trainer.task.inference_step(generator, [trainer.get_model()], adv_sample)
    
    orig_prediction = orig_prediction[0][0]['tokens'].cpu()
    adv_prediction = adv_prediction[0][0]['tokens'].cpu()
    print('Orig Output ', bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(orig_prediction), None)))
    print('Adv Output  ', bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(adv_prediction), None)))    
    if orig_prediction.shape == adv_prediction.shape and all(torch.eq(orig_prediction, adv_prediction)):
        return True
    else:
        return False

def malicious_nonsense(trainer, generator, bpe):
    orig_sample = read_file_input(trainer, bpe)
    if orig_sample is None:
        return None
    adv_sample = read_file_input(trainer, bpe)
    if adv_sample is None:
        return None

    print('Orig Input ', bpe.decode(trainer.task.source_dictionary.string(orig_sample['net_input']['src_tokens'].cpu()[0], None)))
    print('Adv Input  ', bpe.decode(trainer.task.source_dictionary.string(adv_sample['net_input']['src_tokens'].cpu()[0], None)))
    orig_prediction = trainer.task.inference_step(generator, [trainer.get_model()], orig_sample)
    adv_prediction = trainer.task.inference_step(generator, [trainer.get_model()], adv_sample)

    orig_prediction = orig_prediction[0][0]['tokens'].cpu()
    adv_prediction = adv_prediction[0][0]['tokens'].cpu()
    print('Orig Output ', bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(orig_prediction), None)))
    print('Adv Output  ', bpe.decode(trainer.task.source_dictionary.string(torch.LongTensor(adv_prediction), None)))
    if orig_prediction.shape == adv_prediction.shape and all(torch.eq(orig_prediction, adv_prediction)):
        return True
    else:
        return False



parser = options.get_training_parser()
args = options.parse_args_and_arch(parser)
# make sure everything is reset before loading the model
args.reset_optimizer = True
args.reset_meters = True
args.reset_dataloader = True
args.reset_lr_scheduler = True
args.path = args.restore_file
main(args)

