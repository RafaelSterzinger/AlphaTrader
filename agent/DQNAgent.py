from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam, SGD
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
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

        # self.tensorboard = TensorBoard(log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)
        self.model = self._create_model()
        self.from_storage = False
        self.loss = []

    def load_model(self, path):
        self.model = load_model('models/' + path)
        self.from_storage = True

    def _create_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(32, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(8, activation='tanh'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not self.from_storage and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        for i in range(1,4):
            minibatch = random.sample(self.memory, batch_size)
            loss = []

            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    # Q(s',a)
                    target = reward + self.gamma + np.amax(self.model.predict(next_state)[0])

                # Q(s,a)
                target_f = self.model.predict(state)
                # make the agent to approximately map the current state to future discounted reward
                target_f[0][action] = target
                fit = self.model.fit(state, target_f, epochs=1, verbose=0, workers=8, use_multiprocessing=True)
                loss.append(fit.history['loss'])

            # Average loss of episode
            self.loss.append(np.mean([i for j in loss for i in j]))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
