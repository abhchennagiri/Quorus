import gensim.models
import numpy as np
from scipy import spatial
import tensorflow as tf
from nltk import ngrams
import helpers
from nltk.corpus import stopwords

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
NUM_QUESTIONS = 2000

tf.flags.DEFINE_string("training_data_file","./datasets/training.full.tsv","Data source for the training data")

FLAGS = tf.flags.FLAGS

FLAGS._parse_flags()
print("\n Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
            print("{}={}".format(attr.upper(),value))
            print("")                                                                                                                                       

#Load the data
print("Loading data")
q1, q2, y_truth, q1_len, q2_len = helpers.load_data_and_labels(FLAGS.training_data_file)
dataset = list(zip(q1, q2, y_truth))

#q1 = [word for word in q1[0].split() if word not in stopwords.words('english')]


model_file  = "./GoogleNews-vectors-negative300.bin"

model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

def avg_feature_vector(words, model, num_features, index2word_set):
        #function to average all words vectors in a given paragraph
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0

        #list containing names of words in the vocabulary
        #index2word_set = set(model.index2word) this is moved as input param for performance reasons
        for word in words:
            if word in index2word_set:
                nwords = nwords+1
                featureVec = np.add(featureVec, model[word])

        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec



correct = 0
positive_pred = 0
tp = 0
fp = 0
fn = 0

for i in xrange(NUM_QUESTIONS):
    #Remove the stop words
    question = [word for word in q1[i].split() if word not in stopwords.words('english')]
    q1_avg_vector = avg_feature_vector(question, model, 300, model.index2word)
    print ""
    print "---- Number" + str(i) + "----"
    print question

    question = [word for word in q2[i].split() if word not in stopwords.words('english')]
    q2_avg_vector = avg_feature_vector(question, model, 300, model.index2word)
    print question
    sim = 1 - spatial.distance.cosine(q1_avg_vector, q2_avg_vector)
    print "Similarity: ", sim

    label_truth = 1 if y_truth[i][1] == 1 else 0
    label_pred = 1 if sim >= 0.72 else 0

    print "Label truth: ", label_truth
    print "Label_pred: ", label_pred

    print ""

    if label_truth == label_pred:
        correct += 1
        if label_truth == 1:
            tp += 1
        
    if label_pred == 1:
        positive_pred += 1
        if label_truth == 0:
            fp += 1
       
    if label_pred == 0 and label_truth == 1:
            fn += 1

accuracy = ( 1.0 * correct)/(NUM_QUESTIONS)
precision = ( 1.0 * tp )/(tp + fp) if (tp + fp) > 0 else 0
recall = ( 1.0 * tp )/(tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * ((precision * recall))/( precision + recall ) if (precision + recall) > 0 else 0
positive_rate = (1.0 * positive_pred)/(NUM_QUESTIONS)


print accuracy, precision, recall, f1, positive_rate
print "accuracy: %.4f \tprec: %.4f\t recall: %.4f\t f1: %.4f\t pos_rate: %.4f\t" %( accuracy, precision, recall, f1, positive_rate)
