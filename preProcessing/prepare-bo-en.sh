#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

src=bo
tgt=en
lang=bo-en
ROOT=$(dirname "$0")
SCRIPTS=$ROOT/tokenizer
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

BPESIZE=16384
ORIG=$ROOT/data_orig
DATA=$ROOT/data.bo.en.bpe16k
mkdir -p "$DATA"

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

# prepare train, valid and test data 
echo "prepare train, valid, and test data"
cd $ORIG 
for l in $src $tgt; do
    sed -n 1,100p data.$l > ../$DATA/test.$lang.$l 
    sed -n 100,10000p data.$l > ../$DATA/valid.$lang.$l 
    tail -n +10000 data.$l > ../$DATA/train.$lang.$l 
done
cd ..

# learn BPE with sentencepiece
TRAIN_FILES=$(for SRC in src; do echo $DATA/train.$lang.${src}; echo $DATA/train.$lang.${tgt}; done | tr "\n" ",")
echo "learning joint BPE over ${TRAIN_FILES}..."

python "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$DATA/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# encode train, valid, test data 
echo "encoding train, valid, test with learned BPE..."

for i in "valid" "train" "test"; do 
    python "$SPM_ENCODE" \
        --model "$DATA/sentencepiece.bpe.model" \
        --inputs $DATA/$i.$lang.$src $DATA/$i.$lang.$tgt \
        --outputs $DATA/$i.bpe.$lang.$src $DATA/$i.bpe.$lang.$tgt \
        --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
done