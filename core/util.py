import numpy as np
import pandas as pd

# sigmoid function for data normalization
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def load_data(path):
	data = None
	try:
		data = pd.read_csv('../data/' + path)
	except FileNotFoundError:
		print('Could not read file from path.')
	return data

def process_data(data):
	assert data.ndim == 2
	assert data.shape == 7
	assert 'Close' in data
	prices = signal_features = data['Close'].to_numpy()
	return prices, signal_features

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])