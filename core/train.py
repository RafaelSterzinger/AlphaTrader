import matplotlib.pyplot as plt
import tensorflow as tf

from core.util import load_data, process_data, get_logger, get_labeling
from core.env import Env
from agent.DQNAgent import *

logger = get_logger()

# Load data and create environment
df = load_data("AAPL_train.csv")
window_size = 30
frame_bound = (window_size, len(df))
prices, signal_features = process_data(df, window_size, frame_bound)
env = Env(prices, signal_features, df=df, window_size=window_size, frame_bound=frame_bound)

epochs = 650
rewards = []
profits = []

logger.info('env information:')
logger.info('> epochs:', epochs)
logger.info('> max_possible_profit', env.max_possible_profit())

# Create agent
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

logger.info('start training:')
for e in range(epochs):
    # reset state in the beginning of each epoch
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    done = False
    info = None
    epoch_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        epoch_reward += reward

        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)

        # make next_state the new current state for the next episode.
        state = next_state

    # train the agent with the experience of the episode
    agent.replay(32)
    # save sum of rewards of epoch
    rewards.append(env.get_total_reward())
    profits.append(env.get_total_profit())
    assert epoch_reward == env.get_total_reward()

    # plot current epoch with reward and profit
    if e % 130 == 0:
        logger.info('finish epoch ' + str(e))
        logger.info('info: ' + info)

        print('finish epoch ' + str(e))
        print(info)

        plt.cla()
        env.render_all()
        plt.show()

plt.cla()
env.render_all()
plt.savefig("plots/plot" + get_labeling() + ".png")

agent.model.save("models/model_" + get_labeling())
mean_reward = tf.reduce_mean(rewards[-50:])
logger.info("Average of last 50 rewards:" + mean_reward)
print("Average of last 50 rewards:", mean_reward)
