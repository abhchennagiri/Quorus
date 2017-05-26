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
