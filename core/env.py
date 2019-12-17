from gym_anytrading.envs import StocksEnv
from core.util import load_data, process_data, sigmoid


# Load data and create environment
def create_environment(file):
    df = load_data(file)
    window_size = 30
    frame_bound = (window_size, len(df))
    prices, signal_features = process_data(df, window_size, frame_bound)
    return Env(prices, signal_features, df=df, window_size=window_size, frame_bound=frame_bound)


# Custom environment for own defined signal features
class Env(StocksEnv):
    def __init__(self, prices, signal_features, **kwargs):
        self._prices = prices
        self._signal_features = signal_features
        super().__init__(**kwargs)

    def _process_data(self):
        return self._prices, self._signal_features

    def get_total_reward(self):
        return self._total_reward

    def get_total_profit(self):
        return self._total_profit

    # Scale current window with Sigmoid from 0 to 1
    def _get_observation(self):
        current_window = self.signal_features[(self._current_tick - self.window_size):self._current_tick]
        return [[sigmoid(i) for i in j] for j in current_window]

