# Download a model and its test set 
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2
wget https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2
export CUDA_VISIBLE_DEVICES = 0
python simple_attack.py wmt16.en-de.joined-dict.newstest2014/ --arch transformer_vaswani_wmt_en_de_big --restore-file wmt16.en-de.joined-dict.transformer/model.pt  --valid-subset test --bpe subword_nmt --bpe-codes wmt16.en-de.joined-dict.transformer/bpecodes --interactive-attacks