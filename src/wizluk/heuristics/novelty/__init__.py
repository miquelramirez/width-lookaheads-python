# -*- coding: utf-8 -*-

from ._tabular import ClassicTabularNovelty, TabularNovelty, DepthBasedTabularNovelty, DepthBasedTabularNovelty, DepthBasedTabularNoveltyOptimised
from ._base import Feature
from ._state_vars import SVF

_factory_entries = { 'TabularNovelty' : TabularNovelty,\
                    'ClassicTabularNovelty' : ClassicTabularNovelty,\
                    'DepthBasedTabularNovelty': DepthBasedTabularNovelty,\
                    'DepthBasedTabularNoveltyOptimised': DepthBasedTabularNoveltyOptimised
                    }

def create(product_key, **kwargs ) :
    return _factory_entries[product_key](**kwargs)
