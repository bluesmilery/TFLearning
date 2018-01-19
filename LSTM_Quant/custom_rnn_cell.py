#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/16 14:43
# @Author  : Gai
# @File    : custom_rnn_cell.py
# @Contact : plgaixd92498@gmail.com

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl


class SFMCell(rnn_cell_impl.RNNCell):

    def __init__(self, num_units, num_freq, input_size):
        self._num_units = num_units
        self._num_freq = num_freq
        self._input_size = input_size
        self._time = 1

        self.W_sf = tf.Variable(tf.truncated_normal(shape=[self._input_size, self._num_units], stddev=0.01))
        self.U_sf = tf.Variable(tf.truncated_normal(shape=[self._num_units, self._num_units], stddev=0.01))
        self.b_sf = tf.Variable(tf.constant(value=0.01, shape=[self._num_units]))

        self.W_ff = tf.Variable(tf.truncated_normal(shape=[self._input_size, self._num_freq], stddev=0.01))
        self.U_ff = tf.Variable(tf.truncated_normal(shape=[self._num_units, self._num_freq], stddev=0.01))
        self.b_ff = tf.Variable(tf.constant(value=0.01, shape=[self._num_freq]))

        self.W_i = tf.Variable(tf.truncated_normal(shape=[self._input_size, self._num_units], stddev=0.01))
        self.U_i = tf.Variable(tf.truncated_normal(shape=[self._num_units, self._num_units], stddev=0.01))
        self.b_i = tf.Variable(tf.constant(value=0.01, shape=[self._num_units]))

        self.W_c = tf.Variable(tf.truncated_normal(shape=[self._input_size, self._num_units], stddev=0.01))
        self.U_c = tf.Variable(tf.truncated_normal(shape=[self._num_units, self._num_units], stddev=0.01))
        self.b_c = tf.Variable(tf.constant(value=0.01, shape=[self._num_units]))

        self.W_o = tf.Variable(tf.truncated_normal(shape=[self._input_size, self._num_units], stddev=0.01))
        self.U_o = tf.Variable(tf.truncated_normal(shape=[self._num_units, self._num_units], stddev=0.01))
        self.b_o = tf.Variable(tf.constant(value=0.01, shape=[self._num_units]))

        self.u_a = tf.Variable(tf.truncated_normal(shape=[self._num_freq, 1], stddev=0.01))
        self.b_a = tf.Variable(tf.constant(value=0.01, shape=[self._num_units]))
        super().__init__()

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        concat = rnn_cell_impl._concat
        init_Re_s = concat(concat(batch_size, self._num_units), self._num_freq)
        init_Im_s = concat(concat(batch_size, self._num_units), self._num_freq)
        init_h = concat(batch_size, self._num_units)

        # plan-2
        # one_state = [tf.zeros(init_Re_s),
        #              tf.zeros(init_Im_s),
        #              tf.zeros(init_h),
        #              1]

        # plan-3
        one_state = [tf.zeros(init_Re_s),
                     tf.zeros(init_Im_s),
                     tf.zeros(init_h)]

        return one_state

    def __call__(self, inputs, state, name=None):
        Re_s = state[0]
        Im_s = state[1]
        h = state[2]
        # time = state[3] # plan-1/2

        # f_f = frequency forget gate, s_f = state forget gate
        # i = input gate, c = input modulation, o = output gate
        s_f = tf.matmul(inputs, self.W_sf) + tf.matmul(h, self.U_sf) + self.b_sf
        f_f = tf.matmul(inputs, self.W_ff) + tf.matmul(h, self.U_ff) + self.b_ff
        i = tf.matmul(inputs, self.W_i) + tf.matmul(h, self.U_i) + self.b_i
        c = tf.matmul(inputs, self.W_c) + tf.matmul(h, self.U_c) + self.b_c
        o = tf.matmul(inputs, self.W_o) + tf.matmul(h, self.U_o) + self.b_o

        # 使用源码内的实现方法失败，经过一次迭代后输出结果变成nan
        # self._linear = rnn_cell_impl._Linear([inputs, h], 4 * self._num_units + self._num_freq, True)
        # f_f, s_f, i, c, o = tf.split(
        #     value=self._linear([inputs, h]),
        #     num_or_size_splits=[self._num_freq, self._num_units, self._num_units, self._num_units, self._num_units],
        #     axis=1)

        i, c = tf.expand_dims(i, 2), tf.expand_dims(c, 2)
        f_f = tf.reshape(f_f, [-1, 1, self._num_freq])
        s_f = tf.reshape(s_f, [-1, self._num_units, 1])

        # state-frequency forget gate
        # f = tf.matmul(tf.reshape(s_f, [-1, self._num_units, 1]),
        #               tf.reshape(f_f, [-1, 1, self._num_freq]))
        f = s_f * f_f

        # Fourier basis of K frequency components
        omega = np.asarray(2 * np.pi * self._time *
                           np.linspace(1 / self._num_freq, 1, self._num_freq),
                           dtype='float32')
        omega = tf.convert_to_tensor(omega, dtype=tf.float32)
        re = tf.sin(omega)
        im = tf.cos(omega)

        # state-frequency matrix
        new_Re_s = f * Re_s + tf.sigmoid(i) * tf.tanh(c) * re
        new_Im_s = f * Im_s + tf.sigmoid(i) * tf.tanh(c) * im

        # amplitude. phase is ignored
        A = tf.sqrt(tf.square(new_Re_s) + tf.square(new_Im_s))
        new_h = tf.sigmoid(o) * \
                tf.tanh(
                    tf.reshape(
                        tf.matmul(
                            tf.reshape(A, [-1, self._num_freq]),
                            self.u_a),
                        [-1, self._num_units])
                    + self.b_a)

        # new_time = time + 1 # plan-1/2
        self._time += 1  # plan-3

        # plan-1/2
        # return new_h, [new_Re_s, new_Im_s, new_h, new_time]

        # plan-3
        return new_h, [new_Re_s, new_Im_s, new_h]
