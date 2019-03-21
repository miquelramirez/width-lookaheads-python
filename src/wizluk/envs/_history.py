import gym
import wizluk.envs
import wizluk.errors
import copy

from wizluk.utils.obs_filter import MeanStdFilter

class History(object):
    """
        This object wraps OpenAI gym environments and allows
        to associate with them a prefix of either scalar inputs,
        or maps from states to inputs via a policy.
    """
    def __init__(self, env_name, seed = None):
        self.name = env_name
        self._env = gym.make(env_name)
        self._filter = MeanStdFilter(self._env.observation_space.shape)
        self._seed = seed
        self.history = []
        self.policies = []
        self.parent = None
        self.x = None
        self.done = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_env' or k == '_filter':
                continue
            setattr(result, k, copy.deepcopy(v, memo))
        # Now it is guaranteed that the attribute "name" is set
        setattr(result, '_env', gym.make(self.name))
        setattr(result, '_filter', self._filter.copy())
        return result

    @property
    def stats(self):
        return self._filter.get_stats()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def extend(self, pi):
        new_history = History(self.name,self._seed)
        new_history.append_policy(pi)
        new_history.parent = self
        return new_history

    def append_input(self, u):
        self.history.append(lambda e, x : e.step(u) )

    def append_policy(self, pi):
        self.history.append(lambda e, x : e.step(pi.act(x)) )


    def seed(self, v):
        self._env.seed(v)

    def reset(self):
        #if self.seed is not None:
        #    self._env.seed(self.seed)
        self.x = self._env.reset()
        self.x = self.x.flatten()
        for u in self.history:
            self.x, r, self.done, info = u(self._env, self.x)
            self.x = self.x.flatten()
            if self.done: break # as per the gym specs
        self._filter(self.x)

        #return self.x
        return self.stats[0] # return the mean


    def step(self, u):
        return self._env.step(u)
