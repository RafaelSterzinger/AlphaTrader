from core.util import get_labeling, update_performance, visualize_profit, visualize_rewards, visualize_trades
from agent.DQNAgent import *
from core.env import create_environment

def train(data: str):
    env = create_environment(data)

    # Create agent
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

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next episode.
            state = next_state

        # train the agent with the experience of the episode
        agent.replay(32)

        # Save sum of rewards and profit for error metric of epoch
        rewards.append(env.get_total_reward())
        profits.append(env.get_total_profit())

        print(e)
        # Plot current 50th epoch with reward and profit
        if e % 50 == 49:
            print('finish epoch ' + str(e+1))
            print(info)

    # Save model for evaluation
    agent.model.save("models/model_" + get_labeling())

    calc_results_and_update(profits,rewards)

# Calculates performance and updates performance history
def calc_results_and_update(profits, rewards):
    mean_profit = np.mean(profits[-50:])
    print("Average of last 50 profits:", mean_profit)
    visualize_profit(profits)

    mean_reward = np.mean(rewards[-50:])
    print("Average of last 50 rewards:", mean_reward)
    visualize_rewards(rewards)

    update_performance(mean_profit, mean_reward, profits[-1],rewards[-1])
