# -*- coding: utf-8 -*-
# pylint: disable=relative-import
from ._base import Memory, Observation
from .and_or_graph import AND_OR_Memory

_factory_entries = { 'AndOrGraph' : AND_OR_Memory }

def create(product_key, **kwargs ) :
    return _factory_entries[product_key](**kwargs)
