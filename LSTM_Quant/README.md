# LSTM
使用LSTM模型进行股价走势预测，参照这篇文章来进行开发的：http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction

文中使用的Keras，这里使用TensorFlow进行重写。运行run.py即可。

添加了一种名为SFM(State Frequency Memory recurrent network)的网络进行股价走势预测，源代码见：https://github.com/z331565360/State-Frequency-Memory-stock-prediction

源代码使用的是Keras，这里使用TensorFlow进行重写。SFM相关论文见下。

### 版本信息
TensorFlow == 1.4.0

### 文件结构
* run.py：运行模型，负责结果输出
* stock_data.py：使用tushare库获取股票信息
* network_model.py：定义LSTM模型
* custom_rnn_cell.py：自定义RNN Cell，现包括SFM Cell

### 论文阅读
* Zhang L, Aggarwal C, Qi G J. **Stock Price Prediction via Discovering Multi-Frequency Trading Patterns**[C]. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017: 2141-2149.
* Hu H, Qi G J. **State-Frequency Memory Recurrent Neural Networks**[C]. International Conference on Machine Learning. 2017: 1568-1577.