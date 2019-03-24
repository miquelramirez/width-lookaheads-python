# -*- coding: utf-8 -*-

import wizluk
import wizluk.util

class Observation(object) :
    """
        Basic definition of what is to be remembered
    """
    def __init__(self, **kwargs) :
        self._state = kwargs.get('state',None)
        self._action = kwargs.get('action',None)
        self._next_state = kwargs.get('next_state',None)
        self._reward = kwargs.get('reward', None)
        self._is_terminal = kwargs.get('is_terminal', None)

    @property
    def state(self) :
        return self._state

    @property
    def next_state(self) :
        return self._next_state

    @property
    def action(self) :
        return self._action

    @property
    def reward(self) :
        return self._reward

    @property
    def terminal(self) :
        return self._is_terminal



class Memory(object) :
    """
        Base class for "agent" memory component
    """
    def __init__(self, **kwargs) :
        pass

    @property
    def name(self) :
        return self._name

    def remember( self, obs ) :
        """
            Stores obs
        """
        wizluk.util.raise_not_defined()

    def retrieve_batch( self, **batch_params) :
        """
            Produces a "batch", or recollection of memories, according
            as per the given parameters
        """
        wizluk.util.raise_not_defined()
