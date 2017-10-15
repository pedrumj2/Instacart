import pymysql
import numpy as np
import tensorflow as tf


# This class gets data for a single user
class LSTM(object):
    def __init__(self,layers, seed=-1):
        self._input_size = -1
        self._hidden_layer = -1
        self.layers = layers
        self.seed = seed

    def apply(self, input_data, state):
        _conc = LSTM._conc_input(input_data, state)
        self._input_size = np.size(input_data, 1)
        self._hidden_layer = np.size(input_data, 1)
        return self._block(_conc)

    def _block(self, input_data):
        _size = np.size(input_data, 1)
        if self.seed != -1:
            tf_wt = tf.Variable(
                tf.random_normal([self._hidden_layer, _size], stddev=0.35, seed=self.seed))
            tf_wi = tf.Variable(
                tf.random_normal([self._hidden_layer, _size], stddev=0.35, seed=self.seed))
            tf_wc = tf.Variable(
                tf.random_normal([self._hidden_layer, _size], stddev=0.35, seed=self.seed))
            tf_wo = tf.Variable(
                tf.random_normal([self._hidden_layer, _size], stddev=0.35, seed=self.seed))
        else:
            tf_wt = tf.Variable(
                tf.random_normal([self._hidden_layer, _size], stddev=0.35))
            tf_wi = tf.Variable(
                tf.random_normal([self._hidden_layer, _size], stddev=0.35))
            tf_wc = tf.Variable(
                tf.random_normal([self._hidden_layer, _size], stddev=0.35))
            tf_wo = tf.Variable(
                tf.random_normal([self._hidden_layer, _size], stddev=0.35))
        tf_bt = tf.Variable(tf.zeros([self._hidden_layer, 1]))
        tf_bi = tf.Variable(tf.zeros([self._hidden_layer, 1]))
        tf_bc = tf.Variable(tf.zeros([self._hidden_layer, 1]))
        tf_bo = tf.Variable(tf.zeros([self._hidden_layer, 1]))
        tf_ct_1 = tf.Variable(tf.zeros([self._hidden_layer, 1]))

        tf_ft =  tf.nn.sigmoid(tf.matmul(tf_wt, input_data) + tf_bt)
        tf_it = tf.nn.sigmoid(tf.matmul(tf_wi, input_data) + tf_bi)
        tf_chat_t = tf.nn.tanh(tf.matmul(tf_wc, input_data) + tf_bc)
        tf_ct_1 = tf.multiply(tf_ft,tf_ct_1) + tf.multiply(tf_it,tf_chat_t)
        tf_ot = tf.nn.sigmoid(tf.matmul(tf_wo, input_data) + tf_bo)
        return tf.multiply(tf_ot, tf.nn.tanh(tf_ct_1))

    @staticmethod
    def _conc_input(input_data, state_data):
        _output = tf.concat([input_data, state_data], 0)
        return _output


