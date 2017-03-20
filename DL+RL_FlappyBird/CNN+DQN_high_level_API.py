from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

import random
from collections import deque

import skimage as skimage
from skimage import transform, color, exposure


GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1





def build_cnn_model(features, labels, mode):

    #input_layer = tf.placeholder(tf.float32, [None, 80, 80, 4])
    input_layer = tf.reshape(features, [-1, 80, 80, 4])
    conv1 = tf.layers.conv2d(inputs = input_layer,
                             filters = 32,
                             kernel_size = 8,
                             strides = 4,
                             padding = "same",
                             activation = tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs = conv1,
                                    pool_size = 2,
                                    strides = 2)

    conv2 = tf.layers.conv2d(inputs = pool1,
                             filters = 64,
                             kernel_size = 4,
                             strides = 2,
                             padding = "same",
                             activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                    pool_size = 2,
                                    strides = 2)

    conv3 = tf.layers.conv2d(inputs = pool2,
                             filters = 64,
                             kernel_size = 3,
                             strides = 1,
                             padding = "same",
                             activation = tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs = conv3,
                                    pool_size = 2,
                                    strides = 2)

    doflat = tf.reshape(pool3, [-1, 256])

    dense = tf.layers.dense(inputs = doflat,
                            units = 512,
                            activation = tf.nn.relu)

    Qvalue_output = tf.layers.dense(inputs = dense,
                             units = 2)

    loss = None
    train_op = None

    # configure the training op (for train mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss = loss,
                                                   global_step = tf.contrib.framework.get_global_step(),
                                                   learning_rate = 0.001,
                                                   optimizer = "SGD")

    # generate predictions
    predictions = {
        "classes": tf.argmax(input = Qvalue_output, axis = 1)
    }

    # return a modelFnOps object
    return model_fn_lib.ModelFnOps(mode = mode,
                                   predictions = predictions,
                                   loss = loss,
                                   train_op = train_op)


    #return input_layer, Qvalue_output

#def train_network(input_layer, Qvalue_output):
def main(unuse_argv):
    flappyBird = game.GameState()

    D = deque()

    action0 = np.zeros(ACTIONS)
    action0[0] = 1

    observation0, reward0, terminal = flappyBird.frame_step(action0)

    observation0 = skimage.color.rgb2gray(observation0)
    observation0 = skimage.transform.resize(observation0, (80, 80))
    observation0 = skimage.exposure.rescale_intensity(observation0, out_range = (0, 255))

    s_t = np.stack((observation0, observation0, observation0, observation0), axis = 2)

    epsilon = INITIAL_EPSILON
    time = 0
    bird_brain = learn.Estimator(model_fn=build_cnn_model)

    while (True) :
        #Qvalue = Qvalue_output.eval(feed_dict={input_layer: [s_t]})
        #Qvalue = bird_brain.evaluate(input_fn = {input_layer:[s_t]})
        Qvalue = bird_brain.predict(s_t)
        action = np.zeros(ACTIONS)
        action_index = 0

        if time % FRAME_PER_ACTION == 0:
            if random.random <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                action[action_index] = 1
            else:
                action_index = np.argmax(Qvalue)
                action[action_index] = 1
        else:
            action[0] = 1










if __name__ == "__main__":
    # sess = tf.InteractiveSession()
    # input_layer, Qvalue_output = build_cnn_model()
    # train_network(input_layer, Qvalue_output)
    tf.app.run()