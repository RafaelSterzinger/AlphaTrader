import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

from core.util import load_data, process_data, get_labeling, update_performance
from core.env import Env
from agent.DQNAgent import *

# Load data and create environment
df = load_data("AAPL_train.csv")
window_size = 30
frame_bound = (window_size, len(df))
prices, signal_features = process_data(df, window_size, frame_bound)
env = Env(prices, signal_features, df=df, window_size=window_size, frame_bound=frame_bound)

epochs = 5
rewards = []
profits = []

print('env information:')
print('epochs:' + str(epochs))
print('max_possible_profit' + str(env.max_possible_profit()))

# Create agent
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

print('start training:')
for e in range(epochs):
    # reset state in the beginning of each epoch
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    done = False
    info = None

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)

        # make next_state the new current state for the next episode.
        state = next_state

    # train the agent with the experience of the episode
    agent.replay(200)
    # save sum of rewards of epoch
    rewards.append(env.get_total_reward())
    profits.append(env.get_total_profit())

    # plot current epoch with reward and profit
    if e % 50 == 0:
        print('finish epoch ' + str(e))
        print(info)

        plt.cla()
        env.render_all()
        plt.show()

plt.cla()
env.render_all()
plt.savefig("plots/plot" + get_labeling() + ".png")

agent.model.save("models/model_" + get_labeling())

mean_profit = np.mean(profits[-50:])
print("Average of last 50 profits:", mean_profit)

mean_reward = np.mean(rewards[-50:])
print("Average of last 50 rewards:", mean_reward)

update_performance(mean_profit, mean_reward)
