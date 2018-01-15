#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 16:22
# @Author  : Gai
# @File    : run.py
# @Contact : plgaixd92498@gmail.com


import stock_data as sd
import network_model as nm

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_results(predicted_data, true_data):
    """
    绘制预测数据与真实数据
    :param predicted_data: 预测数据
    :param true_data: 真实数据
    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, 'k', label='True Data')

    if predicted_data.shape[0] == true_data.shape[0]:
        plt.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.show()
    else:
        for i, data in enumerate(predicted_data):
            padding = [None for _ in range(i * predicted_data.shape[1])]
            plt.plot(padding + list(data), label='Prediction')
            plt.legend()
        plt.show()


if __name__ == '__main__':
    time_step = 14
    filename = sd.get_stock_data()
    x_train, y_train, x_test, y_test = sd.process_data(filename=filename,
                                                       time_step=time_step,
                                                       train_percentage=0.9,
                                                       is_normalized=True)

    model = nm.LSTM_Model(learning_rate=0.001,
                          time_step=time_step,
                          layers_size=[1, 50, 100, 1])

    model.build_model()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        model.train_process(x=x_train[:, :, np.newaxis],
                            y=y_train[:, np.newaxis],
                            keep_prob=0.8,
                            num_epoch=50,
                            batch_size=128,
                            sess=sess)

        y_predict = model.predict_process(x=x_test[:, :, np.newaxis],
                                          y=y_test[:, np.newaxis],
                                          sess=sess,
                                          type=3)

    plot_results(y_predict, y_test)
