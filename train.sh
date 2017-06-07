#!/bin/bash
#d=50
#d=100
d=100
#d=300

s=.05
#s=.1

lambda=0.0
#lambda=0.1
#lambda=1.0

#set="dev"
set="full"

word_vectors="./glove.6B.${d}d.txt"
#word_vectors="./glove.twitter.27B/glove.twitter.27B.${d}d.txt"
#word_vectors="./Quora_question_pair_partition/wordvec.txt"

training_data="./datasets/training.${set}.tsv"
#training_data="./Quora_question_pair_partition/train.${set}.tsv"

python train.py \
    --training_data_file="${training_data}" \
    --dev_sample_percentage=${s} \
    --batch_size=64 \
    --embeddings_file="${word_vectors}" \
    --embedding_dim=${d} \
    --l2_reg_lambda=${lambda}
