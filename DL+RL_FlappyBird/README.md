# Deep Reinforcement Learning
仿照DeepMind使用Deep Reinforcement Learning玩Atari系列游戏，将其移植到Flappy Bird游戏上。项目使用Python脚本语言进行编程，使用TensorFlow框架开发。参照这篇文章来进行开发的：https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

目前DQN算法实现的版本有：
* **NIPS DQN**(experience replay)
* **Nature DQN**(fixed Q target network parameters)
* **Double DQN**(double Q learning)

### 文件结构
* low level API是使用tf.nn等基础API进行编程
* high level API是使用tf.contrib.learn、tf.layers等更高层封装的API进行编程（未完成）

### 论文阅读
* Mnih V, Kavukcuoglu K, Silver D, et al. **Playing atari with deep reinforcement learning**[J]. arXiv preprint arXiv:1312.5602, 2013.
* Mnih V, Kavukcuoglu K, Silver D, et al. **Human-level control through deep reinforcement learning**[J]. Nature, 2015, 518(7540): 529-533.
* Hasselt H V. **Double Q-learning**[C]. Advances in Neural Information Processing Systems. 2010: 2613-2621.
* Van Hasselt H, Guez A, Silver D. **Deep Reinforcement Learning with Double Q-Learning**[C]. AAAI. 2016: 2094-2100.
