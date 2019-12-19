from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from _collections import deque

import numpy as np
import random


class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size  # input neurons
        self.action_size = action_size  # output neurons
        self.memory = deque(maxlen=20_000)  # replay memory

        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

        self.model = self._create_model()
        self.from_storage = False

    def load_model(self, path: str):
        self.model = load_model('models/' + path)
        self.from_storage = True

    def _create_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state: [[]], action: int, reward: float, next_state: [[]], done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: [[]]):
        # epsilon greedy, for evaluation do not explore
        if not self.from_storage and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        # training the model multiple times per epoch improved performance a lot
        for i in range(1, 4):
            minibatch = random.sample(self.memory, batch_size)

            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    # Q(s',a)
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                # Q(s,a)
                target_f = self.model.predict(state)
                # make the agent to approximately map the current state to future discounted reward
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0, workers=8, use_multiprocessing=True)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
