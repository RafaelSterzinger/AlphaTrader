from core.env import create_environment
from core.util import visualize_trades
from core.agent import DQNAgent
import numpy as np


# test is used to load and test a model
def test(data: str, model: str):
    env = create_environment(data)

    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load_model(model)

    print('start evaluation:')
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        print(info)

    visualize_trades(env, False, model)

