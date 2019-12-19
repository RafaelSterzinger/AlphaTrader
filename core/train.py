from core.util import create_label, visualize_profit, visualize_rewards
from core.agent import *
from core.env import create_environment


# train is used to train and store a model
def train(data: str):
    env = create_environment(data)

    # create agent
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    epochs = 650
    rewards = []
    profits = []

    print('epochs: ' + str(epochs))
    print('start training:')
    for e in range(epochs):
        # reset state in the beginning of each epoch
        state = env.reset()

        # flatten
        state = np.reshape(state, [1, state_size])

        done = False
        info = None

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # stop epoch if agent lost 20%
            if env.get_total_profit() < 0.8:
                break

            # make next_state the new current state for the next episode.
            state = next_state

        # train the agent with the experience of the episode
        agent.replay(64)

        # save sum of rewards and profit for error metric of epoch and plots
        rewards.append(env.get_total_reward())
        profits.append(env.get_total_profit())

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
