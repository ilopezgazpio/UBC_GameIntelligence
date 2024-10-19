import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id = '8Puzzle-v0',
    entry_point = 'gym_8Puzzle.envs:PuzzleEnv',
    max_episode_steps = 1000000000,
    reward_threshold = 1.0,
    nondeterministic = False,
)