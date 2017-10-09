import numpy as np
import tensorflow as tf

from DataMod.User import User
from ResAux.CrossEntropy import CrossEntropy
from ResAux.Results import Results


def run(rnnAux, rnn_length, count_train, hidden_layer):
    print("inside model1")
    predictor_size = 33+User.max_user
    label_size = 49688

    tf_x = tf.placeholder(tf.float32, [rnn_length, predictor_size, 1])
    tf_drop_out_prob = tf.placeholder(tf.float32)
    tf_label = tf.placeholder(tf.float32, [rnn_length, label_size, 2])
    tf_prev_holder = tf.placeholder(tf.float32, [hidden_layer, 1])
    tf_prev = tf_prev_holder

    tf_W = tf.Variable(tf.random_normal([hidden_layer, predictor_size+label_size+hidden_layer], stddev=0.35))
    tf_W1 = tf.Variable(tf.random_normal([hidden_layer,hidden_layer], stddev=0.35))
    tf_U = tf.Variable(tf.random_normal([label_size, hidden_layer, 2], stddev=0.35))
    tf_bu = tf.Variable(tf.random_normal([label_size, 1, 2], stddev=0.35))
    tf_b = tf.Variable(tf.zeros([hidden_layer, 1]))
    tf_b1 = tf.Variable(tf.zeros([hidden_layer, 1]))

    tf_train_steps = []
    tf_cross_entropy = None
    # Initializing the variables

    for i in range(rnn_length):
        tf_prev_x_label = tf.concat([tf_prev, tf_x[i], tf.reshape(tf_label[i, :, 0], (label_size, 1))], 0)
        tf_prev = tf.nn.sigmoid(tf.matmul(tf_W, tf_prev_x_label) + tf_b)
        tf_prev = tf.nn.sigmoid(tf.matmul(tf_W1, tf_prev) + tf_b1)

        tf_res_0 = tf.matmul(tf.nn.dropout(tf_U[:, :, 0], tf_drop_out_prob), tf_prev) + tf_bu[:, :, 0]
        tf_res_1 = tf.matmul(tf.nn.dropout(tf_U[:, :, 1], tf_drop_out_prob), tf_prev) + tf_bu[:, :, 1]
        tf_res = tf.nn.softmax(tf.concat((tf_res_0, tf_res_1), 1), 1)
        #tf_cross_entropy = -tf.reduce_mean(tf_res[:, 0] * tf.log(tf_label[i, :, 0]) +
        #                                                tf_res[:, 1] * tf.log(tf_label[i, :, 1])*4968, reduction_indices=[0])
        #tf_cross_entropy = -tf.reduce_mean(tf.reduce_sum(tf_label[i] * tf.log(tf.clip_by_value(tf_res, 0.00001, 1)), reduction_indices=[1]))
        tf_cross_entropy = -tf.reduce_mean(
              4968*tf_label[i, :, 0] * tf.log(tf.clip_by_value(tf_res[:, 0], 0.00001, 1)) +
                          tf_label[i, :, 1] * tf.log(tf.clip_by_value(tf_res[:, 1], 0.00001, 1)))
        tf_train_step = tf.train.AdamOptimizer().minimize(tf_cross_entropy)
        tf_train_steps.append(tf_train_step)

    init = tf.global_variables_initializer()
    _cross_entropy_helper = CrossEntropy()
    print("done building graph")
    with tf.Session() as sess:
        sess.run(init)
        flag = True
        i = 0

        while i < count_train:
            pred_out, lab_out, is_new = rnnAux.get_training()
            if is_new:
                prev = np.zeros((hidden_layer, 1), np.float32)
            output = sess.run([tf_train_steps, tf_cross_entropy, tf_prev, tf_res, tf_res_0], {tf_x: pred_out, tf_label: lab_out,
                                                                        tf_prev_holder: prev,
                                                                        tf_drop_out_prob: 0.5})
            prev= output[2]
            _cross_entropy_helper.add_value(output[1])
            _cross_entropy_helper.print_res()
            i += 1

        i = 0
        results = Results()
        flag = True
        while flag:
            pred_out, lab_out, is_new, is_end_user, is_end_all = rnnAux.get_test()
            if not is_end_all:
                if is_new:
                    prev = np.zeros((hidden_layer, 1), np.float32)
                output = sess.run([tf_cross_entropy, tf_prev, tf_res, tf_res_0],
                                  {tf_x: pred_out, tf_label: lab_out,
                                   tf_prev_holder: prev,
                                  tf_drop_out_prob: 1.0})
                prev = output[1]
                print(output[0])
                if is_end_user:
                    results.add_results(lab_out[rnn_length-1, :, 0], output[2][:, 0])
            else:
                flag = False
        results.print_output()
        d =     3
    x =3
