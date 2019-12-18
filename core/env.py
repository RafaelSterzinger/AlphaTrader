from gym_anytrading.envs import TradingEnv,Actions, Positions
from core.util import load_data, process_data, sigmoid_scale, min_max_scale, standard_scale


# Load data and create environment
def create_environment(file):
    df = load_data(file)
    window_size = 30
    frame_bound = (window_size, len(df))
    prices, signal_features = process_data(df, window_size, frame_bound)
    return Env(prices, signal_features, df=df, window_size=window_size)


# Custom environment for own defined signal features
class Env(TradingEnv):
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

    # Scale current window with StandardScaler from 0 to 1
    def _get_observation(self):
        current_window = self.signal_features[(self._current_tick - self.window_size):self._current_tick]
        return standard_scale(current_window)

    def _calculate_reward(self, action):
        current_price = self.prices[self._current_tick]
        last_price = self.prices[self._current_tick-1]
        price_diff = current_price - last_price

        if (action == Actions.Buy.value):
            return price_diff
        elif (action == Actions.Sell.value):
            return price_diff*-1

    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = self._total_profit / last_trade_price
                self._total_profit = shares * current_price

            elif self._position == Positions.Short:
                shares = self._total_profit / current_price
                self._total_profit = shares * last_trade_price