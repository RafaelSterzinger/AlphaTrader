import gym
import sys
import matplotlib.pyplot as plt

from core.util import load_data, process_data
from agent.DQNAgent import *

data = load_data("AAPL.csv")
prices, signal_features = process_data(data)

env = gym.make("stocks-v0")
state_size = env.observation_space.shape[0]*env.observation_space.shape[1]
action_size = env.action_space.n
agent = DQNAgent(state_size,action_size)

episodes = 650

print("env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())

rewards = []
for e in range(episodes):
    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    while not done:
        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        # make next_state the new current state for the next frame.
        state = next_state
    print()

    # train the agent with the experience of the episode
    agent.replay(32)

    plt.cla()
    env.render_all()
    plt.show()

