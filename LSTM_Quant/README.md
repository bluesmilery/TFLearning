# LSTM
使用LSTM模型进行股价走势预测，参照这篇文章来进行开发的：http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction

文中使用的Keras，这里使用Tensorflow进行重写。运行run.py即可。

### 文件结构
* run.py：运行模型，负责结果输出
* stock_data.py：使用tushare库获取股票信息
* network_model.py：定义LSTM模型
