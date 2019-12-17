import numpy as np
import pandas as pd
#import logging

from datetime import datetime

# def get_logger():
# 	FORMAT = '%(asctime)s %(message)s'
# 	logging.basicConfig(filename='logs/general/' + str(datetime.now().date()) + '.txt', format=FORMAT, level=logging.INFO)
# 	return logging.getLogger()

def get_labeling():
	date = datetime.now()
	return date.strftime("%d_%H_%M")

# # sigmoid function for data normalization
# def sigmoid(x):
# 	return 1 / (1 + np.exp(-x))

def load_data(path):
	data = None
	try:
		data = pd.read_csv('data/' + path)
	except FileNotFoundError:
		print('Could not read file from path.')
	return data

# Load and update performance history
def update_performance(mean_profit, mean_reward):
	performance = pd.read_csv("logs/performance.csv")
	performance.loc[len(performance)] = [get_labeling(), mean_profit, mean_reward]
	performance.to_csv("logs/performance.csv", index_label=False)


def process_data(df, window_size, frame_bound):
	assert len(frame_bound) == 2
	assert df.ndim == 2
	assert df.shape[1] == 7
	assert 'Close' in df
	start = frame_bound[0] - window_size
	end = frame_bound[1]
	prices = df.loc[:, 'Close'].to_numpy()[start:end]
	signal_features = df.loc[:, ['Close']].to_numpy()[start:end]
	return prices, signal_features

# # returns an an n-day state representation ending at time t
# def getState(data, t, n):
# 	d = t - n + 1
# 	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
# 	res = []
# 	for i in range(n - 1):
# 		res.append(sigmoid(block[i + 1] - block[i]))
#
# 	return np.array([res])