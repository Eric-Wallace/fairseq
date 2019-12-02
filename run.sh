conda activate temp; export CUDA_VISIBLE_DEVICES=0; python dont_flip.py google/ --arch transformer_vaswani_wmt_en_de_big --restore-file checkpoint_best.pt --bpe sentencepiece --sentencepiece-vocab sentencepiece.google.bpe.model 
conda activate temp; export CUDA_VISIBLE_DEVICES=1; python dont_flip.py wmt16.en-de.joined-dict.newstest2014/ --arch transformer_vaswani_wmt_en_de_big --restore-file wmt16.en-de.joined-dict.transformer/model.pt  --valid-subset test --bpe subword_nmt --bpe-codes wmt16.en-de.joined-dict.transformer/bpecodes --interactive_attacks
conda activate temp; export CUDA_VISIBLE_DEVICES=1; python dont_flip.py google --arch transformer_vaswani_wmt_en_de_big --restore-file google.checkpoint.pt   --bpe sentencepiece --sentencepiece-vocab sentencepiece.google.bpe.model --interactive_attacks --targeted_flips


python dont_flip.py wmt16.en-de.joined-dict.newstest2014/ --arch transformer_vaswani_wmt_en_de_big --restore-file wmt16.en-de.joined-dict.transformer/model.pt  --valid-subset test --bpe subword_nmt --bpe-codes wmt16.en-de.joined-dict.transformer/bpecodes --targeted-flips --interactive-attacks --no-check-resegmentation --get-multiple-results

python dont_flip.py wmt16.en-de.joined-dict.newstest2014/ --arch transformer_vaswani_wmt_en_de_big --restore-file wmt16.en-de.joined-dict.transformer/model.pt  --valid-subset test --bpe subword_nmt --bpe-codes wmt16.en-de.joined-dict.transformer/bpecodes --interactive-attacks --no-check-resegmentation --random-start

