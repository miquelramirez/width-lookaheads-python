# pylint: disable=wildcard-import, unused-wildcard-import, unused-import, invalid-name, too-many-instance-attributes, missing-docstring, R0914
"""
    Antishape MDP by John Langford, Microsoft Research, 2018
    Adapted from: https://github.com/JohnLangford/RL_acid
"""

import math
import copy
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding



class Antishape_Env(gym.Env):
    """
        OpenAI gym interface for the "Antishape" MDP
    """

    def __init__(self, **kwargs):
        self.action_space = spaces.Discrete(2)
        self.viewer = None
        self.num_states = int(kwargs.get('num_states'))
        self.horizon = 2 * self.num_states

        self.initState = int(kwargs.get('initState', 0))
        self.initStateFactor = self.num_states / 10.0
        self.initStates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert(self.initState < len(self.initStates))
        self.state = (int(self.initStates[self.initState] * self.initStateFactor), 0) # initial state

        self.observation_space = spaces.Box(low=0.0, high=self.num_states - 1.0, shape=(1,), dtype=np.float32)
        self.dynamics = {}
        for i in range(self.num_states):
            left_state = np.max((0, i - 1))
            right_state = np.min((i+1, self.num_states-1))
            left_reward = 0.25 / (float)(left_state+1)
            right_reward = 0.25 / (float)(right_state+1)
            if right_state == self.num_states-1:
                right_reward = 1.0
            self.dynamics[i] = [(left_state, left_reward), (right_state, right_reward)]

    def seed(self, seed=None):
        seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.state = (int(self.initStates[self.initState] * self.initStateFactor), 0)

    def step(self, action):
        s, t = self.state
        assert t < self.horizon
        next_s, r = self.dynamics[s][action]
        self.state = (next_s, t+1)
        terminal = t+1 == self.horizon
        return next_s, r, terminal, {}

    def render(self, mode='human'):
        pass


    def close(self):
        if self.viewer:
            self.viewer.close()
