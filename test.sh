RUN="1496655187"  # baseline 50d
RUN="1496791226"
RUN="1496795873" #LSTM Single 100d hidden_state = 10 #Acc 77.68
#RUN="1496802201" #LSTM Single 100d hidden_state = 128 #Acc= 77.64

vectors="./glove.6B.100d.txt"
#vectors="./glove.twitter.27B/glove.twitter.27B.200d.txt"

test_data="./datasets/test.full.tsv"
#test_data="./Quora_question_pair_partition/test.tsv"

# tensorboard --logdir ./runs/${RUN}/summaries/
python test.py --test_data_file="${test_data}" --checkpoint_dir="./runs/${RUN}/checkpoints" --embeddings_file="${vectors}"
