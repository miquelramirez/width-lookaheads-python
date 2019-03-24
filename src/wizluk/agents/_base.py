# -*- coding: utf-8 -*-

import wizluk
import wizluk.util

class Agent :
    """
    Agent base class
    """
    def __init__(self, **kwargs ) :
        self._domain = kwargs.get('domain', 'unnamed_domain')
        self._name = kwargs.get('name', 'unnamed_agent')
        self._description = kwargs.get('description', '')

    @property
    def domain(self) :
        return self._domain

    @property
    def name(self):
        return self._name

    @property
    def description(self) :
        return self._description

    @property
    def model_filename_prefix(self) :
        return '{}_{}'.format(self.domain.lower(),self.name.lower())

    def act(self, state, ctx): # for Roboschool
        return self.get_action(state)

    def get_action(self, state) :
        """
        The agent maps a state into an action
        """
        wizluk.util.raise_not_defined()

    def store(self) :
        return dict(domain=self._domain, name=self._name, description=self._description)

    def start_episode(self) :
        pass

    def stop_episode(self) :
        pass
