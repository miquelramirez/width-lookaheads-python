# -*- coding: utf-8 -*-

import sys
import inspect

import os

from collections import Counter
import linecache
import tracemalloc

import wizluk
import wizluk.agents

def raise_not_defined():
    file_name = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: '{}' at line {} of {}}".format(method, line, file_name))
    sys.exit(1)

def load_agent(agent):
    if agent in dir(wizluk.agents) :
        return getattr(wizluk.agents, agent)
    raise RuntimeError('Agent {} not found in package wizluk.agents'.format(agent))

def fix_seed_and_possibly_rerun():
    """
    Make sure the environment variable `PYTHONHASHSEED` is set to 1 so that the order of some of the problem's
    components, which is determined by iterating a Python dictionary, is always consistently the same.

    To do so, (@see http://stackoverflow.com/a/25684784), this method might have to spawn a subprocess
    which is identical to the current process in everything but in its set of environment variables,
    in which case it will return True.

    :return: True iff a new subprocess mirroring the current one was executed.
    """
    # Base case: Seed has already been fixed, so we simply return False to signal that execution can carry on normally
    if get_seed() == 1:
        return False

    # Otherwise we print a warning and re-run the process with a fixed hash seed envvar.
    print('\n' + "*" * 80)
    print("WARNING! Fixing PYTHONHASHSEED to 1 to obtain more reliable results")
    print("*" * 80 + '\n', flush=True)
    # We simply set the environment variable and re-call ourselves.
    import subprocess
    env = dict(os.environ)
    env.update(PYTHONHASHSEED='1')
    try :
        subprocess.call(["python3"] + sys.argv, env=env)
    except FileNotFoundError :
        subprocess.call(["python"] + sys.argv, env=env)
    return True

def get_seed():
    try:
        return int(os.environ['PYTHONHASHSEED'])
    except KeyError as _:
        return None


def mkdirp(directory):
    """" mkdir -p -like functionality """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occured


def save_file(name, content):
    with open(name, "w") as f:
        f.write(content)


def is_int(s):
    if isinstance(s, float) : return False
    try:
        int(s)
        return True
    except (ValueError, TypeError) as e:
        return False

def is_float(s):
    if isinstance(s, int) : return False
    try:
        float(s)
        return True
    except (ValueError, TypeError) as e:
        return False

# Sieve of Eratosthenes
# Code by David Eppstein, UC Irvine, 28 Feb 2002
# http://code.activestate.com/recipes/117119/
def gen_primes():
    """ Generate an infinite sequence of prime numbers.
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}

    # The running integer that's checked for primeness
    q = 2

    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            #
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            #
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]

        q += 1

# Tool for memory profiling in Python
# https://stackoverflow.com/a/45679009/338107
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
