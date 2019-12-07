export CUDA_VISIBLE_DEVICES=0; python dont_flip.py wmt18trans/wmt18ensemble/ --arch transformer_vaswani_wmt_en_de_big --source-lang en --target-lang de --restore-file wmt18trans/wmt18ensemble/wmt18.model1.pt  --valid-subset test --bpe subword_nmt --bpe-codes wmt18trans/wmt18ensemble/code --malicious-appends  --read-file-input > wmt18model1.append.raw
python postprocess_malicious_appends.py wmt18model1.append.raw
export CUDA_VISIBLE_DEVICES=0; python dont_flip.py wmt18trans/wmt18ensemble/ --arch transformer_vaswani_wmt_en_de_big --source-lang en --target-lang de --restore-file wmt18trans/wmt18ensemble/wmt18.model2.pt  --valid-subset test --bpe subword_nmt --bpe-codes wmt18trans/wmt18ensemble/code --malicious-appends  --read-file-input > wmt18model2.append.raw
python postprocess_malicious_appends.py wmt18model2.append.raw
#export CUDA_VISIBLE_DEVICES=0; python dont_flip.py wmt18trans/wmt18ensemble/ --arch transformer_vaswani_wmt_en_de_big --source-lang en --target-lang de --restore-file wmt18trans/wmt18ensemble/wmt18.model3.pt  --valid-subset test --bpe subword_nmt --bpe-codes wmt18trans/wmt18ensemble/code --malicious-appends  --read-file-input > wmt18model3.append.raw
#python postprocess_malicious_appends.py wmt18model3.append.raw
export CUDA_VISIBLE_DEVICES=0; python dont_flip.py wmt16trans/wmt16.en-de.joined-dict.newstest2014/ --arch transformer_vaswani_wmt_en_de_big --restore-file wmt16trans/wmt16.en-de.joined-dict.transformer/model.pt  --valid-subset test --bpe subword_nmt --bpe-codes wmt16trans/wmt16.en-de.joined-dict.transformer/bpecodes --malicious-appends --read-file-input > wmt16model.results
python postprocess_malicious_appends.py wmt16model.append.raw
export CUDA_VISIBLE_DEVICES=0; python dont_flip.py wmt14conv/wmt14.en-de.fconv-py/ --source-lang en --target-lang de --arch fconv_wmt_en_de --restore-file wmt14conv/wmt14.en-de.fconv-py/model.pt --valid-subset test --bpe subword_nmt --bpe-codes wmt14conv/wmt14.en-de.fconv-py/bpecodes --malicious-appends --read-file-input > wmt14convmodel.append.raw
python postprocess_malicious_appends.py wmt14convmodel.append.raw
#export CUDA_VISIBLE_DEVICES=0; python dont_flip.py wmt17conv/wmt14.en-de.fconv-py/ --source-lang en --target-lang de --arch fconv_wmt_en_de --restore-file wmt17conv/wmt14.en-de.fconv-py/model.pt --valid-subset test --bpe subword_nmt --bpe-codes wmt17conv/wmt14.en-de.fconv-py/bpecodes --malicious-appends --read-file-input > wmt17convmodel.append.raw
#python postprocess_malicious_appends.py wmt17convmodel.append.raw

#export CUDA_VISIBLE_DEVICES=0; python dont_flip.py iwslttrans/iwslt14.tokenized.de-en/ --arch transformer_iwslt_de_en --restore-file iwslttrans/checkpoint_best.pt --valid-subset test --bpe subword_nmt --bpe-codes iwslttrans/iwslt14.tokenized.de-en/code --malicious-appends --read-file-input > iwsltmodel.append.raw
#python postprocess_malicious_appends.py iwsltmodel.append.raw
