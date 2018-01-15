#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 16:12
# @Author  : Gai
# @File    : network_model.py
# @Contact : plgaixd92498@gmail.com

import numpy as np
import tensorflow as tf


class LSTM_Model():

    def __init__(self,
                 learning_rate,
                 time_step,
                 layers_size):
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
        self.all_cell_output, self.last_cell_state = tf.nn.dynamic_rnn(cell=multi_lstm_cell, inputs=self.x_input,
                                                                       dtype=tf.float32)

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
                      num_epoch,
                      batch_size,
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

        for epoch in range(num_epoch):
            print("epoch:", epoch + 1)
            for i in range(int(x.shape[0] / batch_size)):
                start_index = i * batch_size
                end_index = start_index + batch_size
                batch_x = x[start_index:end_index]
                batch_y = y[start_index:end_index]
                p_cost, _ = sess.run([cost_function, train],
                                     feed_dict={self.x_input: batch_x,
                                                self.y_label: batch_y,
                                                self.keep_prob: keep_prob})
                print(p_cost)

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
