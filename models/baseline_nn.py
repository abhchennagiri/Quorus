import tensorflow as tf
import numpy as np

class BaselineNN( object ):
    """
    A neural network for text classification.
    Uses an embedding layer, followed by a convolutional max_pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, pretrained_embeddings):
        #Define placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")

        self.input_y = tf.placeholder(tf.float32,[None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.input_x1_len = tf.placeholder(tf.int32, [None], name="input_x1_len")
        self.input_x2_len = tf.placeholder(tf.int32, [None], name="input_x2_len")

        #Embedding Layer
        init_embeddings = tf.Variable( pretrained_embeddings )

        embedded_x1 = tf.nn.embedding_lookup(init_embeddings, self.input_x1)
        embedded_x2 = tf.nn.embedding_lookup(init_embeddings, self.input_x2)

        r1 = tf.reduce_mean( embedded_x1, axis=1 )
        r2 = tf.reduce_mean( embedded_x2, axis=1 )

        features = tf.concat(1, [r1, r2, r1-r2, tf.multiply(r1, r2)])

        logits = tf.contrib.layers.fully_connected( features, num_classes, activation_fn=None )

        pred = tf.nn.softmax( logits )

        #Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.scores = pred
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
                

       #Calculate mean cross entropy loss
       with tf.name_scope("loss"):
           losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
           self.loss = tf.reduce_mean(losses)

      #Accuracy
      with tf.name_scope("accuracy"):
          self.y_truth = tf.argmax(self.input_y, 1, name="y_truth")
          correct_predictions = tf.equal( self.predictions, tf.argmax(self.input_y, 1))
          self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"), name="accuracy")



