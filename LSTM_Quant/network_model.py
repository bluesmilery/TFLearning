#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 16:12
# @Author  : Gai
# @File    : network_model.py
# @Contact : plgaixd92498@gmail.com

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import matplotlib.pyplot as plt

from custom_rnn_cell import SFMCell


class LSTM_Model():

    def __init__(self,
                 learning_rate,
                 time_step,
                 layers_size,
                 batch_size,
                 num_epoch):
        """
        类初始化，一些超参
        :param learning_rate: 学习率
        :param time_step: 时间维度大小
        :param layers_size: 网络层结构
        """
        self.learning_rate = learning_rate
        self.feature_size = layers_size[0]
        self.time_step = time_step
        self.hidden_size = layers_size[1:-1]  # LSTM Cell中unit（神经元）个数
        self.output_size = layers_size[-1]  # 网络输出层神经元个数
        self.batch_size = batch_size
        self.num_epoch = num_epoch

    def build_model(self):
        """
        构建网络模型
        """
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.time_step, self.feature_size])
        self.y_label = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        lstm_cell = []
        for one_hidden_layer_size in self.hidden_size:
            one_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=one_hidden_layer_size)
            one_cell = tf.nn.rnn_cell.DropoutWrapper(cell=one_cell, output_keep_prob=self.keep_prob)
            lstm_cell.append(one_cell)
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cell)

        # 创建init state时使用的batch size为动态的，跟input时的batch大小一致
        # 参照dynamic_rnn源码实现
        # 以下三种实现中，第一、二种可以完美运行，第三种在predict的时候会出现维度不匹配错误
        # 第一种
        flat_input = nest.flatten(self.x_input)
        dynamic_batch_size = tf.shape(flat_input[0])[0]
        init_state = multi_lstm_cell.zero_state(batch_size=dynamic_batch_size,dtype=tf.float32)
        self.all_cell_output, self.last_cell_state = tf.nn.dynamic_rnn(cell=multi_lstm_cell,
                                                                       inputs=self.x_input,
                                                                       initial_state=init_state,
                                                                       dtype=tf.float32)
        #
        # 第二种
        # self.all_cell_output, self.last_cell_state = tf.nn.dynamic_rnn(cell=multi_lstm_cell,
        #                                                                inputs=self.x_input,
        #                                                                dtype=tf.float32)
        #
        # 第三种，网上教程均为此种写法
        # 因为state的大小写死为batch size，所以在predict的时候，样本数量跟self.batch_size不一致时便会报错
        # init_state = multi_lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        # self.all_cell_output, self.last_cell_state = tf.nn.dynamic_rnn(cell=multi_lstm_cell,
        #                                                                inputs=self.x_input,
        #                                                                initial_state=init_state,
        #                                                                dtype=tf.float32)

        W_output = tf.Variable(
            tf.truncated_normal(shape=[self.hidden_size[-1] * self.time_step, self.output_size], stddev=0.01))
        b_output = tf.Variable(tf.constant(value=0.01, shape=[self.output_size]))

        self.y_predict = tf.add(
            tf.matmul(a=tf.reshape(self.all_cell_output, [-1, self.hidden_size[-1] * self.time_step]),
                      b=W_output),
            b_output)

    def train_process(self,
                      x,
                      y,
                      keep_prob,
                      sess):
        """
        训练网络
        :param x: 训练数据的x（特征）
        :param y: 训练数据的y（标签）
        :param keep_prob: Dropout时神经元输出保留的概率
        :param num_epoch: 训练迭代次数
        :param batch_size: 训练batch大小
        :param sess: TensorFlow的Session，用于启动图模型计算
        """
        print("Start: Train")

        cost_function = tf.losses.mean_squared_error(labels=self.y_label,
                                                     predictions=self.y_predict)
        train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost_function)

        sess.run(tf.global_variables_initializer())
        import matplotlib.pyplot as plt
        result = []
        for epoch in range(self.num_epoch):
            print("epoch:", epoch + 1)
            for i in range(int(x.shape[0] / self.batch_size)):
                start_index = i * self.batch_size
                end_index = start_index + self.batch_size
                batch_x = x[start_index:end_index]
                batch_y = y[start_index:end_index]
                p_cost, _ = sess.run([cost_function, train],
                                     feed_dict={self.x_input: batch_x,
                                                self.y_label: batch_y,
                                                self.keep_prob: keep_prob})
                # print(p_cost)
                result.append(p_cost)
        plt.plot(result)
        plt.show()
        print("End: Train")

    def predict_process(self,
                        x,
                        y,
                        sess,
                        type):
        """
        使用网络进行预测
        :param x: 预测数据的x（特征）
        :param y: 预测数据的y（标签）
        :param sess: TensorFlow的Session，用于启动图模型计算
        :param type: 预测类型
        :return:
        """
        print("Start: Predict")

        y_predict = 0
        if type == 1:
            # 逐点预测
            y_predict = sess.run(self.y_predict,
                                 feed_dict={self.x_input: x,
                                            self.keep_prob: 1.0})

        elif type == 2:
            # 只使用最初一个样本来预测整个序列
            current_data = x[0]
            y_predict = []
            for i in range(x.shape[0]):
                y_predict.append(sess.run(self.y_predict,
                                          feed_dict={self.x_input: current_data[np.newaxis, :, :],
                                                     self.keep_prob: 1.0})[0, 0])
                current_data = np.insert(current_data[1:], self.time_step - 1, y_predict[-1], axis=0)

        elif type == 3:
            # 分段预测整个序列
            y_predict = []
            for i in range(int(x.shape[0] / self.time_step)):
                current_data = x[i * self.time_step]
                sub_predict = []
                for _ in range(self.time_step):
                    sub_predict.append(sess.run(self.y_predict,
                                                feed_dict={self.x_input: current_data[np.newaxis, :, :],
                                                           self.keep_prob: 1.0})[0, 0])
                    current_data = np.insert(current_data[1:], self.time_step - 1, sub_predict[-1], axis=0)
                y_predict.append(sub_predict)

        print("End: Predict")

        # TODO：增加准确率计算
        return np.array(y_predict)


class SFM_Model():

    def __init__(self,
                 learning_rate,
                 time_step,
                 layers_size,
                 freq_size,
                 batch_size,
                 num_epoch):

        self.learning_rate = learning_rate
        self.feature_size = layers_size[0]
        self.time_step = time_step
        self.hidden_size = layers_size[1:-1]
        self.output_size = layers_size[-1]
        self.frequency_size = freq_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch

    def build_model(self):
        """
        构建网络模型
        """
        print('Start: Build model')
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.time_step, self.feature_size])
        self.y_label = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])
        self.keep_prob = tf.placeholder(dtype=tf.float32)


        sfm_cell = []
        state = []
        for index, (one_hidden_layer_size, one_frequency_size) in enumerate(zip(self.hidden_size, self.frequency_size)):
            one_cell = SFMCell(num_units=one_hidden_layer_size,
                               num_freq=one_frequency_size,
                               input_size=list([self.feature_size] + self.hidden_size)[index])
            one_cell = tf.nn.rnn_cell.DropoutWrapper(cell=one_cell, output_keep_prob=self.keep_prob)
            sfm_cell.append(one_cell)
            # plan-1
            # one_state = [tf.zeros([self.batch_size, one_hidden_layer_size, one_frequency_size]),
            #              tf.zeros([self.batch_size, one_hidden_layer_size, one_frequency_size]),
            #              tf.zeros([self.batch_size, one_hidden_layer_size]),
            #              1]
            # state.append(one_state)
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=sfm_cell)

        # plan-2/3
        flat_input = nest.flatten(self.x_input)
        dynamic_batch_size = tf.shape(flat_input[0])[0]
        init_state = multi_lstm_cell.zero_state(batch_size=dynamic_batch_size,dtype=tf.float32)

        # plan-3
        self.all_cell_output, self.last_cell_state = tf.nn.dynamic_rnn(cell=multi_lstm_cell,
                                                                       inputs=self.x_input,
                                                                       initial_state=init_state,
                                                                       dtype=tf.float32)
        self.last_cell_output = self.last_cell_state[-1][2]

        # plan-1/2
        # # state = init_state
        # self.all_cell_output = []
        # # with tf.variable_scope("RNN"):
        # for t in range(self.time_step):
        #         # if t > 0: tf.get_variable_scope().reuse_variables()
        #     output, state = multi_lstm_cell(self.x_input[:, t, :], state)
        #     self.all_cell_output.append(output)
        # self.last_cell_output = self.all_cell_output[-1]

        W_output = tf.Variable(
            tf.truncated_normal(shape=[self.hidden_size[-1], self.output_size], stddev=0.01))
        b_output = tf.Variable(tf.constant(value=0.01, shape=[self.output_size]))
        # self.testout = tf.reshape(self.all_cell_output, [-1, self.hidden_size[-1] * self.time_step])
        self.y_predict = tf.add(
            tf.matmul(a=self.last_cell_output,
                      b=W_output),
            b_output)
        print("End: Build Model")

    def train_process(self,
                      x,
                      y,
                      keep_prob,
                      sess):

        print("Start: Train")

        cost_function = tf.losses.mean_squared_error(labels=self.y_label,
                                                     predictions=self.y_predict)
        train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost_function)

        sess.run(tf.global_variables_initializer())
        result = []
        for epoch in range(self.num_epoch):
            print("epoch:", epoch + 1)
            # np.random.shuffle(x)
            for i in range(int(x.shape[0] / self.batch_size)):
                start_index = i * self.batch_size
                end_index = start_index + self.batch_size
                batch_x = x[start_index:end_index]
                batch_y = y[start_index:end_index]

                p_cost, _ = sess.run([cost_function, train],
                                     feed_dict={self.x_input: batch_x,
                                                self.y_label: batch_y,
                                                self.keep_prob: keep_prob})
                # print(p_cost)
                result.append(p_cost)
        print("End: Train")
        plt.plot(result)
        plt.show()
        y_predict = sess.run(self.y_predict,feed_dict={self.x_input:x,
                                                       self.keep_prob:1})
        # # plt.plot(result)
        plt.plot(y)
        plt.plot(y_predict)
        plt.show()

    def predict_process(self,
                        x,
                        y,
                        sess,
                        type):

        print("Start: Predict")

        y_predict = 0
        if type == 1:
            # 逐点预测
            y_predict = sess.run(self.y_predict,
                                 feed_dict={self.x_input: x,
                                            self.keep_prob: 1.0})

        elif type == 2:
            # 只使用最初一个样本来预测整个序列
            current_data = x[0]
            y_predict = []
            for i in range(x.shape[0]):
                y_predict.append(sess.run(self.y_predict,
                                          feed_dict={self.x_input: current_data[np.newaxis, :, :],
                                                     self.keep_prob: 1.0})[0, 0])
                current_data = np.insert(current_data[1:], self.time_step - 1, y_predict[-1], axis=0)

        elif type == 3:
            # 分段预测整个序列
            y_predict = []
            for i in range(int(x.shape[0] / self.time_step)):
                current_data = x[i * self.time_step]
                sub_predict = []
                for _ in range(self.time_step):
                    sub_predict.append(sess.run(self.y_predict,
                                                feed_dict={self.x_input: current_data[np.newaxis, :, :],
                                                           self.keep_prob: 1.0})[0, 0])
                    current_data = np.insert(current_data[1:], self.time_step - 1, sub_predict[-1], axis=0)
                y_predict.append(sub_predict)

        print("End: Predict")

        # TODO：增加准确率计算
        return np.array(y_predict)