from gym_anytrading.envs import StocksEnv
from core.util import process_data, sigmoid

# Custom environment for own defined signal features
class Env(StocksEnv):
    def __init__(self, df, **kwargs):
        prices, signal_features = process_data(df)
        self._prices = prices
        self._signal_features = signal_features
        super().__init__(**kwargs)

    def _process_data(self):
        return self._prices, self._signal_features

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

