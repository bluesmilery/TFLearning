#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 16:19
# @Author  : Gai
# @File    : stock_data.py
# @Contact : plgaixd92498@gmail.com

import csv
import time
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt


def get_stock_data():
    """
    获取股票数据
    :return: 数据文件名，用于下一步处理
    """
    stock_code = '300027'
    start_time = '2009-10-30'
    end_time = '2018-01-01'
    current_time = time.strftime('%Y-%m-%d', time.localtime())
    filename = '%s.csv' % (stock_code)
    qfq_data = ts.get_k_data(code=stock_code,
                             start=start_time,
                             end=current_time,
                             ktype='D',
                             autype='qfq')
    # print(qfq_data)
    qfq_data.to_csv(filename, columns=['close'], header=False, index=False)
    # qfq_data['close'].plot()
    # plt.show()
    return filename


def process_data(filename,
                 time_step,
                 train_percentage,
                 is_normalized):
    """
    对数据进行预处理
    :param filename: 数据文件
    :param time_step: 一条样本的时序长度
    :param train_percentage: 训练数据集占比
    :param is_normalized: 是否将数据进行归一化
    :return: 返回训练数据和测试数据的x和y
    """
    print("Start: Process data")

    csv_reader = csv.reader(open(filename, 'r'))
    origin_data = [float(item[0]) for item in csv_reader]
    processed_data = []
    for index in range(len(origin_data) - time_step):
        processed_data.append(origin_data[index:index + time_step + 1])

    if is_normalized:
        normalized_data = []
        for one_sample in processed_data:
            normalized_data.append([(i / one_sample[0] - 1) for i in one_sample])
        processed_data = normalized_data

    processed_data = np.array(processed_data)
    cut_point = int(train_percentage * processed_data.shape[0])
    train_data = processed_data[:cut_point, :]
    test_data = processed_data[cut_point:, :]
    np.random.shuffle(train_data)
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    print("End: Process data")

    return x_train, y_train, x_test, y_test
