import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.experimental.wrappers import FrameStackObservationV0
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import RecordEpisodeStatistics


def wrap_deepmind(env: gym.Env):
    """Configure environment for DeepMind-style Atari."""
    # Record the statistics of the _underlying_ environment, before frame-skip/reward-clipping/etc.
    env = RecordEpisodeStatistics(env)
    # Standard Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
    )
    env = FrameStackObservationV0(env, 4)
    return env
