#!/bin/bash


src=bo
tgt=en
lang=bo-en
ROOT=$(dirname "$0")
PATH_TO_DATA=$ROOT/preProcessing/data.bo.en.bpe16k

echo "Preprocessing the data in ${PATH_TO_DATA} ... "
fairseq-preprocess --source-lang bo --target-lang en \
    --trainpref $PATH_TO_DATA/train.bpe.bo-en \
    --validpref $PATH_TO_DATA/valid.bpe.bo-en \
    --testpref  $PATH_TO_DATA/test.bpe.bo-en \
    --destdir data-bin/tokenized.bo.en.bpe32k \
    --workers 10 \
    --scoring bleu

echo "Training the data for ${lang} ... "
fairseq-train data-bin/tokenized.bo.en.bpe32k/ \
--max-epoch 50 \
--ddp-backend=no_c10d \
--arch transformer \
--share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr 5e-7 --lr-scheduler inverse_sqrt \
--warmup-updates 4000 --warmup-init-lr '1e-07' \
--label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
--dropout 0.3 --weight-decay 0.0001 \
--save-dir checkpoints \
--max-tokens 900 \
--update-freq 8 \
--tensorboard-logdir tensor \
--scoring bleu

echo "Generate the model on test data of 100 sentences"
fairseq-generate data-bin/tokenized.bo.en.bpe16k/\
    --path checkpoints/checkpoint_best.pt \
    --batch-size 8 --beam 5 --remove-bpe=sentencepiece 


echo "an Interactive intreface to test the model"
fairseq-interactive data-bin/tokenized.bo.en.bpe16k/ \
    --source-lang bo --target-lang en \
    --path checkpoints/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe=sentencepiece

echo "Scarebleu score on the test data"
sacrebleu -t data -l ${lang} --echo src > wmt14-bo-en.src
