# -*- coding: utf-8 -*-

from ._base import Feature
import numpy as np

class SVF(Feature) :
    """
        SVF: values of State Variables as Features
    """

    def __init__(self, index ) :
        super(SVF,self).__init__('x_{}'.format(index), \
                                                    1, \
                                                    lambda s : s[index])
        self._index = index

    def __hash__(self) :
        return hash(self._index)

    @staticmethod
    def create_features( N ) :
        return list(SVF(k) for k in range(N))
