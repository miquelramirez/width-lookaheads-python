# -*- coding: utf-8 -*-

import wizluk
import wizluk.util
import time
from ._base import Agent

class LookaheadAgent(Agent) :

    def __init__( self, env, lookahead, **kwargs ) :
        super(LookaheadAgent,self).__init__(**kwargs)
        self._lookahead = lookahead
        self.env = env
        self.episode_count = 0
        self.episode_runtime = 0.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.runtime = 0.0
        self.accum_runtime = 0.0

    def get_action(self, state) :
        t0 = time.perf_counter()
        selected = self._lookahead.get_action(self.env, state )
        tf = time.perf_counter()
        self.episode_runtime += tf - t0
        return selected

    def get_action_base(self, state, base_policy) :
        t0 = time.perf_counter()
        selected = self._lookahead.get_action(self.env, state, base_policy)
        tf = time.perf_counter()
        self.episode_runtime += tf - t0
        return selected

    def observe_transition(self, state, action, reward, next_state, is_terminal, learning):
        """
            Processes a given transition (s,a,r,s') plus a Boolean flag
            that is true whenever s' is a terminal state. Lookahead agents
            only note what was the reward obtained
        """
        self.episode_reward += reward
        self.step_count += 1

    def store(self) :
        repo = super(LookaheadAgent,self).store()
        repo['statistics'] = {}
        repo['statistics']['episode_count'] = self.episode_count
        repo['statistics']['accum_runtime'] = self.accum_runtime
        return repo

    def start_episode(self) :
        self.episode_reward = 0.0
        self.accum_runtime = 0.0
        self.step_count = 0
        self.episode_runtime = 0.0

    def stop_episode(self) :
        self.episode_count += 1
        self.accum_runtime += self.episode_runtime

    def init_evaluation_statistics(self, df) :
        df['reward'] = []
        df['runtime'] = []
        df['episode'] = []
        df['steps'] = []
        self._lookahead.init_statistics(df)

    def collect_evaluation_statistics( self, df, s0 ) :
        df['reward'] += [ self.episode_reward ]
        df['runtime'] += [ self.episode_runtime ]
        df['episode'] += [ self.episode_count ]
        df['steps'] += [ self.step_count ]
        self._lookahead.collect_statistics(df)
        self._lookahead.reset_statistics()
