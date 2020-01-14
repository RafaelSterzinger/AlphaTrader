from core.util import create_label, visualize_profit, visualize_rewards
from core.agent import *
from core.env import create_environment
from core.train import play
import os


# train is used to train and store a model
def train_general():
    envs = []
    data = []
    for filename in os.listdir("data/general"):
        data.append(filename)
        envs.append(create_environment("general/" + filename))

    # create agent
    state_size = envs[0].observation_space.shape[0] * envs[0].observation_space.shape[1]
    action_size = envs[0].action_space.n
    agent = DQNAgent(state_size, action_size)

    epochs = 650 * len(envs)
    # to delete
    rewards = []
    profits = []

    print('epochs: ' + str(epochs))
    print('start training:')
    for e in range(epochs):
        rand = random.randrange(len(envs))
        env = envs[rand]

        print('using environment of:', data[rand])
        print('starting epoch:',e)
        info = play(agent,env)

        # print final reward and profit of epoch
        print(info)

    # save model for evaluation
    agent.model.save("models/model_" + create_label())

    calc_results_and_update(profits, rewards)


# calculates performance of last 50 epochs
def calc_results_and_update(profits: [float], rewards: [float]):
    mean_profit = np.mean(profits[-50:])
    print("Average of last 50 profits:", mean_profit)
    visualize_profit(profits)

    mean_reward = np.mean(rewards[-50:])
    print("Average of last 50 rewards:", mean_reward)
    visualize_rewards(rewards)
