# -*- coding: utf-8 -*-
__version__ = '0.1.0'
__release__ = 'v0.1.0-alpha'


import logging
import copy
from logging import FileHandler, StreamHandler
import wizluk

debug = False
logger = None


class Struct:
    """
        Class whose instances attributes are constructed at runtime
        from a given dictionary 'entries'. Useful as a container of
        sets of parameters, avoiding to access them via the dictionary
        interface.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

def setup_logger(filename, debug=False):

    logging_level = logging.INFO
    if debug :
        wizluk.debug = debug
        logging_level = logging.DEBUG

    log_filename = '{}'.format(filename)
    print("Logging to file: {}".format(log_filename))
    #logging.basicConfig( #filename = log_filename, \
    #                    level = logging_level,\
    #                    format='%(asctime)s [%(levelname)s] - %(message)s' )
    wizluk.logger = logging.getLogger(__name__)
    wizluk.logger.setLevel(logging_level)
    wizluk.logger.addHandler( FileHandler(log_filename, mode='w') )
    wizluk.logger.addHandler( StreamHandler() )
    for handler in wizluk.logger.handlers :
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))
