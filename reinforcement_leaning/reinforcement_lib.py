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
    """Replay memory

       METHODS
           add    -- Add transition to memory.
           sample -- Sample minibatch from memory.
    """

    def __init__(self, states, actions, size=1000000):
        """Creates a new replay memory.

           Memory(states, action) creates a new replay memory for storing
           transitions with `states` observation dimensions and `actions`
           action dimensions. It can store 1000000 transitions.

           Memory(states, actions, size) additionally specifies how many
           transitions can be stored.
        """

        self.s = np.ndarray([size, states])
        self.a = np.ndarray([size, actions])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, states])
        self.done = np.ndarray([size, 1])
        self.n = 0

    def __len__(self):
        """Returns the number of transitions currently stored in the memory."""

        return self.n

    def add(self, s, a, r, sp, done):
        """Adds a transition to the replay memory.

           Memory.add(s, a, r, sp, done) adds a new transition to the
           replay memory starting in state `s`, taking action `a`,
           receiving reward `r` and ending up in state `sp`. `done`
           specifies whether the episode finished at state `sp`.
        """

        self.s[self.n, :] = s
        self.a[self.n, :] = a
        self.r[self.n, :] = r
        self.sp[self.n, :] = sp
        self.done[self.n, :] = done
        self.n += 1

    def sample(self, size):
        """Get random minibatch from memory.

        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        idx = np.random.randint(0, self.n, size)

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.done[idx]


class Network:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __ilshift__(self, other):
        """
            Copies network weights.

            network2 <<= network1 copies the weights from `network1` into `network2`. The
            networks must have the same structure.
        """

        if isinstance(self, DQN) or isinstance(self, Model):
            self.__model.set_weights(other.__model.get_weights())

        return self

    def combine(self, s, a, force=False):
        """
            Combines state and action vectors into single network input.

            m, reshape = Network.combine(s, a) has five cases. In all cases,
            `m` is a matrix and `reshape` is a shape to which the network Q output
            should be reshaped. The shape will be such that states are in
            rows and actions are in columns of `m`.

            1) `s` and `a` are vectors. They will be concatenated.
            2) `s` is a matrix and `a` is a vector. `a` will be replicated for
               each `s`.
            3) `s` is a vector and `a` is a matrix. `s` will be replicated for
               each `a`.
            4) `s` and `a` are matrices with the same number of rows. They will
               be concatenated.
            5) `s` and `a` are matrices with different numbers of rows or
               force=True. Each `s` will be replicated for each `a`.

            EXAMPLE
               >>> print(network.combine([1, 2], 5))
               (array([[1., 2., 5.]], dtype=float32), (1, 1))
               >>> print(network.combine([[1, 2], [3, 4]], 5))
               (array([[1., 2., 5.],
                       [3., 4., 5.]], dtype=float32), (2, 1))
               >>> print(network.combine([1, 2], [5, 6])) # single action only
               (array([[1., 2., 5.],
                       [1., 2., 6.]], dtype=float32), (1, 2))
               >>> print(network.combine([1, 2], [[5], [6]]))
               (array([[1., 2., 5.],
                      [1., 2., 6.]], dtype=float32), (1, 2))
               >>> print(network.combine([[1, 2], [3, 4]], [5, 6])) # single action only
               (array([[1., 2., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 1))
               >>> print(network.combine([[1, 2], [3, 4]], [[5], [6]]))
               (array([[1., 2., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 1))
               >>> print(network.combine([[1, 2], [3, 4]], [[5], [6]], force=True))
               (array([[1., 2., 5.],
                       [1., 2., 6.],
                       [3., 4., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 2))
        """

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


class DQN(Network):
    """
    Deep learning-based Q approximator.

       METHODS
           train        -- Train network.
           __call__     -- Evaluate network.
           save_model   -- Save trained model.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25], model_name='generic', load_model=False):
        """
        Creates a new Q approximator.

        DQN(states, actions) creates a Q approximator with `states`
        observation dimensions and `actions` action dimensions. It has
        two hidden layers with 25 neurons each. All layers except
        the last use ReLU activation."

        DQN(states, actions, hiddens) additionally specifies the
        number of neurons in the hidden layers.

        DQN(states, actions, hiddens, load_model=True) additionally
        specifies if a previously saved model will be loaded.

        EXAMPLE
           >>> dqn = DQN(2, 1, [10, 10])
        """

        super(DQN, self).__init__(states, actions)

        self.load_model = load_model
        self.model_name = model_name

        if load_model:
            self.__model = tf.keras.models.load_model(model_name)
            self.__model.compile(loss=tf.keras.losses.MeanSquaredError(),
                                 optimizer=tf.keras.optimizers.Adam())

        else:
            inputs = tf.keras.Input(shape=(states + actions,))
            layer = inputs
            for h in hiddens:
                layer = tf.keras.layers.Dense(h, activation='relu')(layer)
            outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

            self.__model = tf.keras.Model(inputs, outputs)
            self.__model.compile(loss=tf.keras.losses.MeanSquaredError(),
                                 optimizer=tf.keras.optimizers.Adam())



    def train(self, s, a, target):
        """
        Trains the Q approximator.

           DQN.train(s, a, target) trains the Q approximator such that
           it approaches DQN(s, a) = target.

           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a vector specifying an
           action for every observation in the batch. `target` is a vector
           specifying a target value for each observation-action pair in
           the batch.

           EXAMPLE
               >>> dqn = DQN(2, 1)
               >>> dqn.train([[0.1, 2], [0.4, 3], [0.2, 5]], [-1, 1, 0], [12, 16, 19])
        """

        self.__model.train_on_batch(self.combine(s, a), np.atleast_1d(target))

    def __call__(self, s, a):
        """
        Evaluates the Q approximator.

           DQN(s, a) returns the value of the approximator at observation
           `s` and action `a`.

           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           If `a` is the same size as the number of rows in `s`, it specifies
           the action at which to evaluate each observation in the batch.
           Otherwise, it specifies the action(s) at which the evaluate ALL
           observations in the batch.

           EXAMPLE
               >>> dqn = DQN(2, 1)
               >>> # single observation and action
               >>> print(dqn([0.1, 2], -1))
               [[ 12 ]]
               >>> # batch of observations and actions
               >>> print(dqn([[0.1, 2], [0.4, 3]], [-1, 1]))
               [[12]
                [16]]
               >>> # evaluate single observation at multiple actions
               >>> print(dqn([0.1, 2], [-1, 1]))
               [[12  -12]]
        """

        inp, reshape = self.combine(s, a)
        return np.reshape(np.asarray(self.__model(inp)), reshape)

    def __ilshift__(self, other):
        """
        Copies network weights.

           network2 <<= network1 copies the weights from `network1` into `network2`. The
           networks must have the same structure.
        """

        self.__model.set_weights(other.__model.get_weights())

        return self

    def save_model(self):
        """
        Saves a trained model
        """
        self.__model.save(self.model_name)
