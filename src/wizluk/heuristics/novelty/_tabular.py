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
