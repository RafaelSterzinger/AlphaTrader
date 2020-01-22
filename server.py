from flask import Flask, jsonify, request
from flask_cors import CORS

from core.agent import DQNAgent
import numpy as np
import pandas as pd

# configuration
from core.env import Env
from core.util import process_data

DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

agent = DQNAgent(30, 2, True)
agent.load_model('general')


@app.route('/', methods=['POST'])
def get_prediction():
    data = request.get_json()
    df = pd.DataFrame(data=data[1:-1],  # values
                      columns=data[0])
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Adj Close'] = df['Adj Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    profit, points = predict(df)
    data = {"profit": profit, "points": points}
    return jsonify(data)


def predict(df):
    window_size = 30
    frame_bound = (window_size, len(df))
    prices, signal_features = process_data(df, window_size, frame_bound)
    env = Env(prices, signal_features, df=df, window_size=window_size)

    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]

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

    return env.get_total_profit(), env.get_positions()


if __name__ == '__main__':
    app.run()
