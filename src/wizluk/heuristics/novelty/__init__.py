# -*- coding: utf-8 -*-

from ._tabular import ClassicTabularNovelty, TabularNovelty, DepthBasedTabularNovelty
from ._base import Feature
from ._state_vars import SVF

_factory_entries = { 'TabularNovelty' : TabularNovelty,\
                    'ClassicTabularNovelty' : ClassicTabularNovelty,\
                    'DepthBasedTabularNovelty': DepthBasedTabularNovelty
                    }

def create(product_key, **kwargs ) :
    return _factory_entries[product_key](**kwargs)
