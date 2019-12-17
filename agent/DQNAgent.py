from datetime import datetime
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard
from _collections import deque

import numpy as np
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # input neurons
        self.action_size = action_size  # output neurons
        self.memory = deque(maxlen=20_000)  # replay memory

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.tensorboard = TensorBoard(log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)
        self.model = self._create_model()

    def _create_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=SGD(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma + np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, verbose=0, workers=8, use_multiprocessing=True)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


