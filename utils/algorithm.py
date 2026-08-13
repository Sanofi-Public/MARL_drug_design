"""
Algorithm class for A2C algorithms.
Code from :https://github.com/uoe-agents/MATE
"""

from abc import ABC, abstractmethod
from gymnasium.spaces.utils import flatdim
from utils.utils import flatten


class Algorithm(ABC):
    def __init__(
        self,cfgparams,
       # observation_spaces,
       # action_spaces,
       # algorithm_config,
       # task_emb_dim,
    ):


        # set all values from config as attributes
        for k, v in flatten(cfgparams).items():
            setattr(self, k, v)