import numpy as np
import random
from wizluk.errors import WizlukError
from gym.spaces import Box

class RandomPolicy(object) :

    def __init__(self, **kwargs) :
        self.name = 'Random'
        self.num_selected = 0

    def get_action(self, env, state):
        legal = [ a for a in range(env.action_space.n) ]
        if len(legal) == 0:
            return None
        self.num_selected += 1
        return random.choice(legal)

    def filter_and_select_action( self, agent, state, candidates ) :
        return random.choice(candidates)

    def reset_statistics(self) :
        self.num_selected = 0

    def init_statistics(self, df) :
        df['num_selected'] = []

    def collect_statistics(self, df ) :
        df['num_selected'] += [ self.num_selected ]
