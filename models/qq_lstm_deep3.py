import tensorflow as tf
import numpy as np

class LSTMDeep3( object ):
    """
    Basic Model - Simple BiLSTM to the question embeddings
    """
    def __init__(self, sequence_length, num_classes, pretrained_embeddings, l2_reg_lambda ):
        #Define the placeholders
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.input_x1_length = tf.placeholder(tf.int32, [None], name="input_x1_length")
        self.input_x2_length = tf.placeholder(tf.int32, [None], name="input_x2_length")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

        #Add L2 regularization here
        l2_loss = tf.constant(0.0)

        #Embeddings Layer
        init_embeddings = tf.Variable( pretrained_embeddings )
        embedding_size = pretrained_embeddings.shape[1]

        embedded_x1 = tf.nn.embedding_lookup( init_embeddings, self.input_x1 )
        embedded_x2 = tf.nn.embedding_lookup( init_embeddings, self.input_x2 )


        r1 = self.biLSTM( embedded_x1, self.input_x1_length, sequence_length, False)
        r2 = self.biLSTM( embedded_x2, self.input_x2_length, sequence_length, True)

        with tf.name_scope("output"):
            features = tf.concat(1, [r1, r2])
            #num_filters_total = num_filters * len(filter_sizes)
            feature_length = 2 * r1.get_shape().as_list()[1]

            #num_hidden1 = 128  #int(np.sqrt(feature_length))
            #num_hidden2 = 256  #int(np.sqrt(feature_length))

            W3= tf.get_variable(
                "W3",
                shape=[feature_length, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b3")

            l2_loss += tf.nn.l2_loss(W3)
            l2_loss += tf.nn.l2_loss(b3)

            self.scores= tf.nn.xw_plus_b(features, W3, b3, name="hidden")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
 
            """
            W4 = tf.get_variable(
                "W4",
                shape=[num_hidden1, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b4")
            """
            

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.y_truth = tf.argmax(self.input_y, 1, name="y_truth")
            self.correct_predictions = tf.equal(self.predictions, self.y_truth, name="correct_predictions")
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy") 
        


    def biLSTM(self, embeddings, x_lengths, sequence_length, reuse):
        with tf.variable_scope("tower", reuse = reuse):
            with tf.name_scope("lstm"):
                x = embeddings

                #Initial state of the LSTM cell memory
                #state_size = 10
                
                # Define lstm cells with tensorflow
                # Forward direction cell
                #lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, forget_bias=1.0, state_is_tuple=True)
                # Backward direction cell
                #lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, forget_bias=1.0, state_is_tuple=True)

                state_size = 10
                num_layers = 3
                cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True) 
                outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    dtype=tf.float32,
                    sequence_length=x_lengths,
                    inputs=x)
                
                output_fw, output_bw = outputs
                states_fw, states_bw = states

                encoded = tf.stack([output_fw, output_bw], axis=3)
                encoded = tf.concat(3, encoded)
                encoded = tf.reshape(encoded, [-1, sequence_length * state_size * 2])

            # Add dropout
            with tf.name_scope("dropout1"):
                h_drop = tf.nn.dropout(encoded, self.dropout_keep_prob)
        return h_drop



