import logging
from gymnasium.envs.registration import register
from copy import deepcopy
from . import datasets

logger = logging.getLogger(__name__)

register(
    id='StockTrading-v0',
    entry_point='gym_StockTrading.envs:StockTradingEnv',
    max_episode_steps=20000,
    reward_threshold=1.0,
    nondeterministic= False,
    kwargs={
        'stock_data': deepcopy(datasets.STOCKS_AAPL),
    }
)