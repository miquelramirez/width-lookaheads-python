# -*- coding: utf-8 -*-
import abc
from abc import ABC

class Feature(object) :

    def __init__(self, name, order, evaluator ) :
        self._name = name
        self._order = order
        self._eval = evaluator
        self._hash_key = hash((self.name, self.order))

    @property
    def name(self) :
        return self._name

    @property
    def f(self) :
        return self._eval

    @property
    def order(self) :
        return self._order

    def __call__(self, obs ) :
        return self._eval(obs)

    def __hash__(self) :
        return self._hash_key


class NoveltyMeasurement(ABC) :

    def __init__(self, **kwargs ) :
        pass

    @abc.abstractmethod
    def process( self, obs ) :
        pass
