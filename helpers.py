import numpy as np
import re
import itertools
import os
from collections import Counter
from collections import OrderedDict
import logging


def clean_str( string, lower=True ):
    """
    Tokenization/string cleaning for the datasets.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #Remove anything that is not a character, a digit, a bracket etc. from the string
    string = re.sub( r"[^A-Za-z0-9(),!?\'\`]"," ", string )
    #Add a space between the apostrophes so that they are not counted as a single word
    string = re.sub( r"\'s"," 's", string )
    string = re.sub( r"\'ve", " 've", string )
    string = re.sub( r"\'re", " 're", string )
    string = re.sub( r"\'d", " 'd", string )
    string = re.sub( r"\'ll", " 'll", string )
    string = re.sub( r"n\'t", " n't", string )
    string = re.sub( r",", " , ", string )
    string = re.sub( r"!", " ! ", string )
    string = re.sub( r"\(", " ( ", string )
    string = re.sub( r"\)", " ) ", string )
    string = re.sub( r"\?", " ? ", string )
    string = re.sub( r"\s{2,}"," ", string )

    result = string.strip()
    if lower:
        result = result.lower()

    return result

def load_data_and_labels( training_data_file ):
    """
    Split the data into words and generate labels.
    Return the split sentences and labels.
    """
    training_data = list(open(training_data_file,"r").readlines())
    training_data = [s.strip() for s in training_data]
    q1 = []
    q2 = []
    labels = []
    q1_len = []
    q2_len = []
    c = 10
    for line in training_data:
        fields = line.split('\t')
        q1_length = len(fields[1].split())
        q2_length = len(fields[2].split())

        if q1_length > 59 or q2_length > 59:
            continue
        q1.append(fields[1].lower())
        q2.append(fields[2].lower())
    
        labels.append([0,1] if fields[0] == '1' else [1, 0])
        q1_len.append(q1_length)
        q2_len.append(q2_length)

    labels = np.concatenate([labels], 0)
    q1_len = np.concatenate([q1_len], 0)
    q2_len = np.concatenate([q2_len], 0)

    return [q1, q2, labels, q1_len, q2_len]


def normalize( word ):
    """
    Normalize the words that have numbers or casing
    """
    return word.lower()


def load_word_vector_mapping( embeddings_file ):
    """
    Load the embedding vector for each word onto the ordered dictionary and return
    """
    ret = OrderedDict()
    for row in list(open(embeddings_file,"r").readlines()):
        fields = row.strip().split()
        word = fields[0]
        ret[word] = np.array(list(map(float, fields[1:])))

    return ret


def load_embeddings(embeddings_file, vocab_dict, embedding_dim, use_cache=True):
    embeddings_cache_file = embeddings_file + ".cache.npy"
    if use_cache and os.path.isfile(embeddings_cache_file):
        embeddings = np.load(embeddings_cache_file)
        return embeddings

    embeddings = np.array(np.random.randn(len(vocab_dict)+ 1, embedding_dim), dtype=np.float32)
    embeddings[0] = 0
    for word, vec in load_word_vector_mapping(embeddings_file).items():
        word = normalize(word)
        if word in vocab_dict:
            embeddings[vocab_dict[word]] = vec
   
    np.save(embeddings_cache_file, embeddings)
    print "Initialized embeddings"
    #print embeddings
    return embeddings

def batch_iter(data, batch_size, shuffle=True):
    """
    Generates batch iterator for a dataset
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1)/batch_size) + 1

    #Shuffle the data at each epoch
    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffled_indices]

    else:
        shuffled_data = data

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1)* batch_size, data_size)
        yield shuffled_data[start_index:end_index]
