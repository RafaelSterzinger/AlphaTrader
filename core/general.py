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

    print('epochs: ' + str(epochs))
    print('start training:')
    for e in range(epochs):
        rand = random.randrange(len(envs))
        env = envs[rand]

        print('using environment of:', data[rand])
        print('starting epoch:', e)
        info = play(agent, env)

        # print final reward and profit of epoch
        print(info)

    # save model for evaluation
    agent.model.save("models/model_" + create_label())
