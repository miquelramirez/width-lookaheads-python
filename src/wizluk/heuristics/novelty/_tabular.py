# -*- coding: utf-8 -*-

from ._base import NoveltyMeasurement, Feature
import numpy as np
#import wizluk
class TabularNovelty(NoveltyMeasurement):

    def __init__(self ) :
        self._tables = {}
        self._ranks = {}
        self._max_rank = 0

    def add_feature(self, f : Feature):
        """
            Registers the feature and initialises the feature-based memory to
            an empty set
        """
        self._tables[f] = set()
        try :
            self._ranks[f.order].append(f)
        except KeyError :
            self._ranks[f.order] = [ f ]
        self._max_rank = max(self._max_rank, f.order)

    def F(self, key) :
        return self._tables[key]

    @property
    def max_rank(self) :
        return self._max_rank

    def process(self, s ) :
        novelty = self.max_rank + 1
        for k in range(1,self.max_rank+1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                old_card = len(table)
                table.add( f(s) )
                if novelty > k and len(table) > old_card :
                    novelty = min(k,novelty)
        return novelty

    def reset(self):
        for _, features in self._ranks.items():
            for f in features:
                self._tables[f] = set()

class ClassicTabularNovelty(TabularNovelty):

    MAX_DEPTH = 2**20 # arbitrary big integer number

    def __init__(self ) :
        super(ClassicTabularNovelty,self).__init__()
        self._depth = {}


    def add_feature(self, f : Feature):
        super(ClassicTabularNovelty,self).add_feature(f)

    def is_novel(self, obs) :
        s = obs
        for k in range(1, self.max_rank+1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                v = f(s)
                if v not in table :
                    return True
        return False

    def update_feature_table(self, obs) :
        s = obs
        for k in range(1,self.max_rank+1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                v = f(s)
                table.add( v )


class DepthBasedTabularNovelty(TabularNovelty):

    MAX_DEPTH = 2**20 # arbitrary big integer number

    def __init__(self ) :
        super(DepthBasedTabularNovelty,self).__init__()
        self._depth = {}

    def reset(self):
        super(DepthBasedTabularNovelty,self).reset()
        self._depth = {}

    def add_feature(self, f : Feature):
        super(DepthBasedTabularNovelty,self).add_feature(f)

    def get_novel_feature(self, obs) :
        s, d = obs
        for k in range(1, self.max_rank+1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                v = f(s)
                if v in table :
                    try :
                        old_d = self._depth[(f,v)]
                    except KeyError :
                        old_d = self.MAX_DEPTH
                    if d <= old_d :
                        return f, v, k, old_d
                else :
                    return f, v, k, self.MAX_DEPTH

        any_feature = self._ranks[1][0]
        try :
            old_d = self._depth[(any_feature, any_feature(s))]
        except KeyError :
            old_d = self.MAX_DEPTH
        return any_feature, any_feature(s), 1, old_d

    def update_feature_table(self, obs) :
        s, d = obs
        for k in range(1,self.max_rank+1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                old_card = len(table)
                v = f(s)
                table.add( v )
                if len(table) == old_card :
                    old_d = self._depth[(f,v)]
                    self._depth[(f,v)] = min(d, old_d )
                else :
                    self._depth[(f,v)] = d

class DepthBasedTabularNoveltyOptimised(TabularNovelty): #Only for depth novelty and when features are state variables for width = 1

    MAX_DEPTH = 2**20 # arbitrary big integer number

    def __init__(self ) :
        super(DepthBasedTabularNoveltyOptimised,self).__init__()
        self._depth = {}
        self._tables = []

    def reset(self):
        for f in range(len(self._tables)):
            self._tables[f] = set()
        self._depth = {}

    def add_feature(self, f : Feature):
        self._tables.append(set())

    def get_novel_feature(self, obs) :
        s, d = obs
        for sIndex in range(len(s)) :
            table = self._tables[sIndex]
            v = s[sIndex]
            if v in table :
                try :
                    old_d = self._depth[(sIndex,v)]
                except KeyError :
                    old_d = self.MAX_DEPTH
                if d <= old_d :
                    return sIndex,v, 1, old_d
            else :
                return sIndex, v, 1, self.MAX_DEPTH

        any_feature = 0
        try :
            old_d = self._depth[(any_feature, s[any_feature])]
        except KeyError :
            old_d = self.MAX_DEPTH
        return any_feature, s[any_feature], 1, old_d

    def update_feature_table(self, obs) :
        s, d = obs
        for sIndex in range(len(s)) :
            table = self._tables[sIndex]
            v = s[sIndex]
            if v in table :
                old_d = self._depth[(sIndex,v)]
                if d < old_d:
                    self._depth[(sIndex,v)] = d
            else :
                table.add(v)
                self._depth[(sIndex,v)] = d

class MaxExpectedRewardBasedTabularNovelty(TabularNovelty):

    #SO: is there something else we can reward to be more general? Like
    #    metric
    MIN_REWARD = -2**20 # arbitrary big negative integer number
    MAX_DEPTH = 2**20 # arbitrary big integer number

    def __init__(self ) :
        super(MaxExpectedRewardBasedTabularNovelty,self).__init__()
        self._accumulated_reward = {}

    def add_feature(self, f : Feature):
        super(MaxExpectedRewardBasedTabularNovelty,self).add_feature(f)

    def get_novel_feature(self, obs) :
        s, V, accumulated_reward, depth = obs
        for k in range(1, self.max_rank+1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                v = f(s)
                if v in table :
                    try :
                        old_V, old_accumulated_reward, old_depth, old_s = self._accumulated_reward[(f,v)]
                    except KeyError :
                        old_V = self.MIN_REWARD
                        old_accumulated_reward = self.MIN_REWARD
                        old_depth = self.MAX_DEPTH
                        old_s = s
                    if V + accumulated_reward > old_V + old_accumulated_reward :
                        #pnl.logger.debug("condition 1")
                        return f, v, k, old_V, old_accumulated_reward, old_depth, old_s
                    elif np.array_equal(s, old_s) and accumulated_reward == old_accumulated_reward and depth == old_depth :
                        #pnl.logger.debug("condition 2")
                        return f, v, k, old_V, old_accumulated_reward, old_depth, old_s
                else :
                    #pnl.logger.debug("condition 3")
                    return f, v, k, self.MIN_REWARD, self.MIN_REWARD, self.MAX_DEPTH, s

        any_feature = self._ranks[1][0]
        try :
            old_V, old_accumulated_reward, old_depth, old_s = self._accumulated_reward[(any_feature, any_feature(s))]
        except KeyError :
            old_V = self.MIN_REWARD
            old_accumulated_reward = self.MIN_REWARD
            old_depth = self.MAX_DEPTH
            old_s = s

        return any_feature, any_feature(s), 1, old_V, old_accumulated_reward, old_depth, old_s

    def update_feature_table(self, obs) :
        s, V, accumulated_reward, depth = obs
        for k in range(1,self.max_rank + 1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                old_card = len(table)
                v = f(s)
                table.add( v )
                if len(table) == old_card :
                    old_V, old_accumulated_reward, old_depth, old_s = self._accumulated_reward[(f,v)]
                    if (V + accumulated_reward > old_V + old_accumulated_reward) :
                        self._accumulated_reward[(f,v)] = V, accumulated_reward, depth, s
                    elif np.array_equal(s, old_s) and accumulated_reward == old_accumulated_reward and depth == old_depth :
                        self._accumulated_reward[(f,v)] = V, accumulated_reward, depth, s
                else :
                    self._accumulated_reward[(f,v)] = V, accumulated_reward, depth, s

class MaxAccumulatedRewardBasedTabularNovelty(TabularNovelty):

    #SO: is there something else we can reward to be more general? Like
    #    metric
    MIN_REWARD = -2**20 # arbitrary big negative integer number
    MAX_DEPTH = 2**20 # arbitrary big integer number
    def __init__(self ) :
        super(MaxAccumulatedRewardBasedTabularNovelty,self).__init__()
        self._accumulated_reward = {}

    def add_feature(self, f : Feature):
        super(MaxAccumulatedRewardBasedTabularNovelty,self).add_feature(f)

    def get_novel_feature(self, obs) :
        s, accumulated_reward = obs
        for k in range(1, self.max_rank+1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                v = f(s)
                if v in table :
                    try :
                        old_accumulated_reward = self._accumulated_reward[(f,v)]
                    except KeyError :
                        old_accumulated_reward = self.MIN_REWARD

                    if accumulated_reward >= old_accumulated_reward :
                        #pnl.logger.debug("condition 1")
                        return f, v, k, old_accumulated_reward
                else :
                    #pnl.logger.debug("condition 3")
                    return f, v, k, self.MIN_REWARD

        any_feature = self._ranks[1][0]
        try :
            old_accumulated_reward = self._accumulated_reward[(any_feature, any_feature(s))]
        except KeyError :
            old_accumulated_reward = self.MIN_REWARD

        return any_feature, any_feature(s), 1, old_accumulated_reward

    def update_feature_table(self, obs) :
        s, accumulated_reward = obs
        for k in range(1,self.max_rank + 1) :
            for f in self._ranks[k] :
                table = self._tables[f]
                old_card = len(table)
                v = f(s)
                table.add( v )
                if len(table) == old_card :
                    old_accumulated_reward = self._accumulated_reward[(f,v)]
                    if (accumulated_reward > old_accumulated_reward) :
                        self._accumulated_reward[(f,v)] = accumulated_reward
                else :
                    self._accumulated_reward[(f,v)] = accumulated_reward
