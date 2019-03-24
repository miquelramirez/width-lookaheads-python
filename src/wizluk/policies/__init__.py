# -*- coding: utf-8 -*-
from ._iw_rollout import IW_Rollout
from ._UCT import UCT
from ._one_step import One_Step
from ._random import RandomPolicy

_factory_entries = {'IW_Rollout' : IW_Rollout,
                    'UCT' : UCT,
                    'RandomPolicy' : RandomPolicy,
                    'One_Step': One_Step,

 }

def create(product_key, **kwargs ) :
    return _factory_entries[product_key](**kwargs)
