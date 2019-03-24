# -*- coding: utf-8 -*-

import numpy as np

class FeatureSet(object):

    def __init__(self, dim):
        self.d = dim
        self.active = []
        self.values = np.zeros(self.d)

    def update(self, s, a):
        pass
