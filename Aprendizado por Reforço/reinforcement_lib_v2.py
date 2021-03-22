"""
Reinforcement Learning Library
Deep Q-Learning

Adalberto Oliveira
Veículos Autônomos - 2021.1
ver. 1.0
"""

# importing libraies
from math import pi
import numpy as np
import scipy.stats
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class Memory:

    def __init__(self, states, actions, size=1000000):

        self.s = np.ndarray([size, states])
        self.a = np.ndarray([size, actions])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, states])
        self.done = np.ndarray([size, 1])
        self.n = 0

    def __len__(self):

        return self.n

    def add(self, s, a, r, sp, done):

        self.s[self.n, :] = s
        self.a[self.n, :] = a
        self.r[self.n, :] = r
        self.sp[self.n, :] = sp
        self.done[self.n, :] = done
        self.n += 1

    def sample(self, size):

        idx = np.random.randint(0, self.n, size)

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.done[idx]


class Network:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __ilshift__(self, other):

        if isinstance(self, DQN) or isinstance(self, Model):
            self.__model.set_weights(other.__model.get_weights())

        return self

    def combine(self, s, a, force=False):

        # Convert scalars to vectors
        s = np.atleast_1d(np.asarray(s, dtype=np.float32))
        a = np.atleast_1d(np.asarray(a, dtype=np.float32))

        
        # Convert vectors to matrices for single-state environments
        if self.states == 1 and len(s.shape) == 1 and s.shape[0] > 1:
            s = np.atleast_2d(s).transpose()

        # Convert vectors to matrices for single-action environments
        if self.actions == 1 and len(a.shape) == 1 and a.shape[0] > 1:
            a = np.atleast_2d(a).transpose()

        # Normalize to matrices
        s = np.atleast_2d(s)
        a = np.atleast_2d(a)

        # Sanity checking
        if len(s.shape) > 2 or len(a.shape) > 2:
            raise ValueError("Input dimensionality not supported")

        if s.shape[1] != self.states:
            raise ValueError("State dimensionality does not match network")

        if a.shape[1] != self.actions:
            raise ValueError("Action dimensionality does not match network")

        # Replicate if necessary
        if s.shape[0] != a.shape[0] or force:
            reshape = (s.shape[0], a.shape[0])
            s = np.repeat(s, np.repeat(reshape[1], reshape[0]), axis=0)
            a = np.tile(a, (reshape[0], 1))
        else:
            reshape = (s.shape[0], 1)
        

        m = np.hstack((s, a))

        return m, reshape

    def combine2(self, s, a, force=False):

        # Convert scalars to vectors
        s = np.atleast_1d(np.asarray(s, dtype=np.float32))
        a = np.atleast_1d(np.asarray(a, dtype=np.float32))     

        m = np.hstack((s, a))

        return m



class DQN(Network):

    def __init__(self, states, actions=1, hiddens=[25, 25], model_name='generic', load_model=False):

        super(DQN, self).__init__(states, actions)

        self.load_model = load_model
        self.model_name = model_name
        
        
        if load_model:
          print('Loading previously created model...')
          self.__model = tf.keras.models.load_model(model_name+".h5")
          self.__model.compile(loss=tf.keras.losses.MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam())

        
        else:
          print('Creating model...')
        
          self.__model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hiddens[0],activation='relu',input_shape=(states+actions,)),
            tf.keras.layers.Dense(hiddens[1],activation='relu'),
            tf.keras.layers.Dense(1,activation='linear')])

          self.__model.compile(loss=tf.keras.losses.MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam())

        self.__model.summary()

        
    def train(self, s, a, target):

        self.__model.train_on_batch(self.combine(s, a), np.atleast_1d(target))

    def train2(self, s, a, target):

        self.__model.train_on_batch(self.combine2(s, a), np.atleast_1d(target))


    def __call__(self, s, a):

        inp, reshape = self.combine(s, a)
        return np.reshape(np.asarray(self.__model(inp)), reshape)

    def __ilshift__(self, other):

        self.__model.set_weights(other.__model.get_weights())

        return self

    def save_model(self):
        self.__model.save(self.model_name+".h5")
