import tensorflow as tf
from nltk import ngrams
import numpy as np
import helpers

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


tf.flags.DEFINE_string("training_data_file","./datasets/training.full.tsv","Data source for the training data")

n_grams = [1, 2, 3]

def jaccard_similarity(s1, s2):
    return (len(s1.intersection(s2)) * 1.0)/ len(s1.union(s2))

FLAGS = tf.flags.FLAGS
print dir(FLAGS)
#print FLAGS.__getattr__()

FLAGS._parse_flags()
print("\n Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(),value))
print("")

#Load the data
print("Loading data")
q1, q2, y_truth, q1_len, q2_len = helpers.load_data_and_labels(FLAGS.training_data_file)
dataset = list(zip(q1, q2, y_truth))
print dataset[:10]

best_t = 0
best_accuracy = 0
c = 10

for t in np.arange(0.05, 0.35, 0.01):
    correct = 0
    positive_pred = 0
    tp = 0
    fp = 0
    fn = 0
    for data in dataset:
        x1,x2,y = data

        s1 = set()
        s2 = set()
        for n in n_grams:
            for n_word in ngrams(x1.split(), n):
                s1.add(n_word)
            for n_word in ngrams(x2.split(), n):
                s2.add(n_word)


        j = jaccard_similarity(s1, s2)
        label_truth = 1 if y[1] == 1 else 0
        label_pred = 1 if j >=t else 0

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

    accuracy = ( 1.0 * correct)/len(dataset)
    precision = ( 1.0 * tp )/(tp + fp) if (tp + fp) > 0 else 0
    recall = ( 1.0 * tp )/(tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * ((precision * recall))/( precision + recall ) if (precision + recall) > 0 else 0
    positive_rate = (1.0 * positive_pred)/len(dataset)
    if accuracy > best_accuracy:
        best_t = t
        best_accuracy = accuracy

    print "t: %.2f \taccuracy: %.4f \tprec: %.4f\t recall: %.4f\t f1: %.4f\t pos_rate: %.4f\t best_t: %.2f\t best_acc: %.4f" %(t, accuracy, precision, recall, f1, positive_rate, best_t, best_accuracy)        


