from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

import random
from collections import deque

import skimage as skimage
from skimage import transform, color, exposure
import cv2
from matplotlib import pyplot as plt

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 1000. # timesteps to observe before training
EXPLORE = 300000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 256 # size of minibatch
FRAME_PER_ACTION = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

# input layer
x_image = tf.placeholder(tf.float32, shape = [None, 80, 80, 4])

# first convolution layer
W_conv1 = weight_variable([8, 8, 4, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 4) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolution layer
W_conv2 = weight_variable([4, 4, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# third convolution layer
W_conv3 = weight_variable([3, 3, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# reshape
h_pool3_flat = tf.reshape(h_pool3, [-1, 2 * 2 * 64])

# first fully connected layer
W_fc1 = weight_variable([2 * 2 * 64, 512])
b_fc1 = bias_variable([512])

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# drop
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer
W_fc2 = weight_variable([512, 2])
b_fc2 = bias_variable([2])

Q_output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


action_for_train = tf.placeholder(tf.float32, shape = [None, ACTIONS])
Q_target = tf.placeholder(tf.float32, shape = [None])
Q_value_for_train = tf.reduce_sum(tf.multiply(Q_output, action_for_train), axis = 1)
cost_function = tf.reduce_mean(tf.square(Q_target - Q_value_for_train))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cost_function)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



flappyBird = game.GameState()

D = deque()

action0 = np.zeros(ACTIONS)
action0[0] = 1

observation0, reward0, terminal = flappyBird.frame_step(action0)

# observation0 = skimage.color.rgb2gray(observation0)
# observation0 = skimage.transform.resize(observation0, (80, 80))
# observation0 = skimage.exposure.rescale_intensity(observation0, out_range = (0, 255))
# state0 = np.stack((observation0, observation0, observation0, observation0), axis = 2)

observation0 = cv2.cvtColor(observation0, cv2.COLOR_BGR2GRAY)
observation0 = cv2.resize(observation0, (80, 80), interpolation = cv2.INTER_AREA)
ret, observation0 = cv2.threshold(observation0, 10, 255, cv2.THRESH_BINARY)
state0 = np.stack((observation0, observation0, observation0, observation0), axis = 2)

# plt.imshow(observation0,'gray')
# plt.show()


state_current = state0

epsilon = INITIAL_EPSILON
time = 0

score = 0
max_score = 0
f = open("score.txt", 'a')
f.write("\n")

while (True):

    Q_value = Q_output.eval(feed_dict = {x_image: [state_current], keep_prob: 1.0})
    action = np.zeros(ACTIONS)
    action_index = 0

    if time % FRAME_PER_ACTION == 0:
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            action[action_index] = 1
        else:
            action_index = np.argmax(Q_value)
            action[action_index] = 1
    else:
        action[action_index] = 1

    if epsilon > FINAL_EPSILON and time > OBSERVATION:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    observation, reward, terminal = flappyBird.frame_step(action)

    # observation = skimage.color.rgb2gray(observation)
    # observation = skimage.transform.resize(observation, (80, 80))
    # observation = skimage.exposure.rescale_intensity(observation, out_range=(0, 255))
    # observation = np.reshape(observation, (80, 80, 1))
    # state_next = np.append(state_current[:, :, 1:], observation, axis = 2)

    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (80, 80), interpolation = cv2.INTER_AREA)
    ret, observation = cv2.threshold(observation, 10, 255, cv2.THRESH_BINARY)
    # plt.imshow(observation, 'gray')
    # plt.show()
    observation = np.reshape(observation, (80, 80, 1))
    state_next = np.append(state_current[:, :, 1:], observation, axis = 2)


    if reward == -1:
        f.write(str(score) + ",")
        if score > max_score:
            max_score = score
        score = 0
    if reward == 1:
        score += 1





    D.append((state_current, action, reward, state_next, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    if time > OBSERVATION:
        minibatch = random.sample(D, BATCH)

        state_current_batch = []
        action_batch = []
        reward_batch = []
        state_next_batch = []
        terminal_batch = []

        for i in range(0, len(minibatch)):
            state_current_batch.append(minibatch[i][0])
            action_batch.append(minibatch[i][1])
            reward_batch.append(minibatch[i][2])
            state_next_batch.append(minibatch[i][3])
            terminal_batch.append(minibatch[i][4])

        Q_value_batch = Q_output.eval(feed_dict = {x_image: state_next_batch, keep_prob: 1.0})
        Q_target_batch = []
        for i in range(0, len(minibatch)):
            if terminal_batch[i]:
                Q_target_batch.append(reward_batch[i])
            else:
                Q_target_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        #test = tf.multiply(Q_output, a)
        # print (Q_value_batch)
        # print (Q_target_batch)
        # print (sess.run(Q_value_for_train, feed_dict={x_image: state_current_batch, action_for_train:action_batch}))
        # print (sess.run(cost_function, feed_dict={x_image: state_current_batch, action_for_train:action_batch, Q_target: Q_target_batch}))
        train_step.run(feed_dict = {x_image: state_current_batch, action_for_train: action_batch, Q_target: Q_target_batch, keep_prob: 0.5})
        # print ("train")



    state_current = state_next
    time += 1

    # print info
    state = ""
    if time <= OBSERVATION:
        state = "observe"
    elif time > OBSERVATION and time <= OBSERVATION + EXPLORE:
        state = "explore"
    else:
        state = "train"

    print("TIMESTEP", time, "/ STATE", state, \
          "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
          "/ Q_MAX %e" % np.max(Q_value), "/ MAX_SCORE", max_score, "SCORE", score)
