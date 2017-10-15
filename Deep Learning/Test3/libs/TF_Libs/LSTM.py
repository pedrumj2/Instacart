import pymysql
import numpy as np
import tensorflow as tf


# This class gets data for a single user
class LSTM(object):
    def __init__(self,layers, seed=-1):
        self._input_size = -1
        self._hidden_layer_size = -1
        self.layers = layers
        self.seed = seed
        self._init_tensors()

    def _init_tensors(self):
        _size = self._hidden_layer_size + self._input_size
        if self.seed != -1:
            self.tf_wt = tf.Variable(
                tf.random_normal([self._hidden_layer_size, _size], stddev=0.35, seed=self.seed))
            self.tf_wi = tf.Variable(
                tf.random_normal([self._hidden_layer_size, _size], stddev=0.35, seed=self.seed))
            self.tf_wc = tf.Variable(
                tf.random_normal([self._hidden_layer_size, _size], stddev=0.35, seed=self.seed))
            self.tf_wo = tf.Variable(
                tf.random_normal([self._hidden_layer_size, _size], stddev=0.35, seed=self.seed))
        else:
            self.tf_wt = tf.Variable(
                tf.random_normal([self._hidden_layer_size, _size], stddev=0.35))
            self.tf_wi = tf.Variable(
                tf.random_normal([self._hidden_layer_size, _size], stddev=0.35))
            self.tf_wc = tf.Variable(
                tf.random_normal([self._hidden_layer_size, _size], stddev=0.35))
            self.tf_wo = tf.Variable(
                tf.random_normal([self._hidden_layer_size, _size], stddev=0.35))
        self.tf_bt = tf.Variable(tf.zeros([self._hidden_layer_size, 1]))
        self.tf_bi = tf.Variable(tf.zeros([self._hidden_layer_size, 1]))
        self.tf_bc = tf.Variable(tf.zeros([self._hidden_layer_size, 1]))
        self.tf_bo = tf.Variable(tf.zeros([self._hidden_layer_size, 1]))
        self.tf_ct_1 = tf.Variable(tf.zeros([self._hidden_layer_size, 1]))

    def apply(self, input_data, state):
        _conc = LSTM._conc_input(input_data, state)
        self._input_size = input_data.get_shape().as_list()[0]
        self._hidden_layer_size = state.get_shape().as_list()[0]
        return self._block(_conc)

    def _block(self, input_data):
        tf_ft =  tf.nn.sigmoid(tf.matmul(self.tf_wt, input_data) + self.tf_bt)
        tf_it = tf.nn.sigmoid(tf.matmul(self.tf_wi, input_data) + self.tf_bi)
        tf_chat_t = tf.nn.tanh(tf.matmul(self.tf_wc, input_data) + self.tf_bc)
        tf_ct_1 = tf.multiply(tf_ft,self.tf_ct_1) + tf.multiply(tf_it,tf_chat_t)
        tf_ot = tf.nn.sigmoid(tf.matmul(self.tf_wo, input_data) + self.tf_bo)
        return tf.multiply(tf_ot, tf.nn.tanh(tf_ct_1))

    @staticmethod
    def _conc_input(input_data, state_data):
        _output = tf.concat([input_data, state_data], 0)
        return _output


