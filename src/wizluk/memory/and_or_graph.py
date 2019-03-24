# -*- coding: utf-8 -*-

import copy
import typing
import random

from wizluk.memory import Memory

class OR_Node(object) :

    def __init__(self, s, d, terminal=False) :
        self._state = copy.deepcopy(s)
        self._d = d
        self._parents = set()
        self._children = {}
        self._terminal = terminal

    @property
    def state(self) :
        return self._state

    @property
    def d(self) :
        return self._d

    @property
    def children(self) :
        return self._children

    @property
    def child(self) :
        return self._children

    @property
    def parents(self) :
        return self._parents

    def add_parent(self, n ) :
        self._parents.add(n)

    @property
    def terminal(self) :
        return self._terminal

    @terminal.setter
    def terminal(self,value):
        self._terminal = value

    def __str__(self):
        return 's: {} d: {} terminal: {}'.format(self.state,self.d,self.terminal)

    def __hash__(self) :
        return hash(bytes(self.state))^hash(self.d)

    def random_child(self) :
        k = random.choice(list(self.children.keys()))
        return self.children[k]

    def __eq__(self, other) :
        return bytes(self.state) == bytes(other.state) and \
            self.d == other.d

class AND_Node(object) :

    def __init__(self, a, parent : OR_Node ) :
        self._parent = parent
        self._parent.children[a] = self
        self._action = a
        self._children = set()
        self._visits = 0

    @property
    def state(self) :
        return self._parent.state

    @property
    def d(self) :
        return self._parent.d

    @property
    def action(self) :
        return self._action

    @property
    def children(self) :
        return self._children

    @property
    def parent(self) :
        return self._parent

    def update_visits(self) :
        self._visits += 1

    def update(self, R : float, succ : OR_Node ) :
        self.update_visits()
        card = len(self._children)
        self._children.add( (succ, R) )
        return len(self._children) != card

    def random_child(self) :
        return random.choice(list(self.children))


    def __hash__(self) :
        return hash((self.action, bytes(self._parent.state), self._parent.d))

    def __eq__(self, other) :
        return bytes(self._parent.state) == bytes(other._parent.state) and \
            self._parent.d == other._parent.d and self._action == other._action


class AND_OR_Graph(object) :

    def __init__(self) :
        self._roots = {}
        self._or_node_index = {} # this may be quite inefficient
        self._and_node_index = {}

    @property
    def roots(self) :
        return self._roots

    @property
    def size(self) :
        return len(self._and_node_index)

    def locate(self, node: OR_Node ) :
        return self._or_node_index[node]

    def register(self, node : OR_Node ) :
        self._or_node_index[node] = node

    def add_root(self, node : OR_Node ) :
        self._roots[node] = node

    def extend(self, node : OR_Node, action, reward, succ : OR_Node ) :
        and_node = AND_Node(action, node)
        node.children[action] = and_node
        and_node.update(reward, succ)
        self._and_node_index[and_node] = and_node
        return and_node

    def update( self, node: OR_Node, action, reward, succ : OR_Node ) :
        #assert node.d == succ.d + 1
        try :
            or_parent = self.locate(node)
        except KeyError :
            self.register(node)
            or_parent = node
        if len(or_parent.parents) == 0 :
            self._roots[or_parent] = or_parent
        try :
            or_succ = self.locate(succ)
        except KeyError :
            self.register(succ)
            or_succ = succ
        try :

            and_node = or_parent.child[action]
            and_node.update(reward, or_succ)
        except KeyError :
            and_node = self.extend( or_parent, action, reward, or_succ )
            and_node.visited = True
            and_node.num_visits = 0
            and_node.Q = float('-inf')
        or_succ.add_parent(and_node)

    def random_or_node(self ) :
        return self._or_node_index[random.choice(list(self._or_node_index.keys()))]

    def random_root_node(self) :
        return self._roots[random.choice(list(self._roots.keys()))]

    def sample_or_nodes(self, N ) :
        selected = random.sample(list(self._or_node_index.keys()),N)
        return [self._or_node_index[k] for k in selected]

class AND_OR_Memory(Memory) :

    def __init__(self, **kwargs ) :
        super(AND_OR_Memory,self).__init__(**kwargs)
        self._capacity = int(kwargs.get('max_memory_states', 2000))
        self._name = 'AndOrGraph'
        self._horizon = int(kwargs.get('horizon',500))
        self._graph = AND_OR_Graph()

    @property
    def G(self) :
        return self._graph

    @property
    def H(self) :
        return self._horizon

    @property
    def capacity(self) :
        return self._capacity

    @property
    def size(self) :
        return self.G.size

    def  __len__(self) :
        return self.G.size

    def remember( self, obs ) :
        p = OR_Node(obs.state,self.H)
        if obs.terminal :
            k = 0
        else :
            k = self.H
        q = OR_Node(obs.next_state,k)
        self.G.update(p, obs.action, obs.reward, q )

    def retrieve_batch(self, **batch_params) :
        batch_size = int(batch_params.get('batch_size',32))
        batch = []
        while batch_size > 0 :
            s = self.G.random_root_node()
            # construct batch as random walk
            while s.d > 0 and batch_size > 0 and len(s.children) > 0:
                a = s.random_child()
                t, r = a.random_child()
                terminal = t.d == 0
                batch.append( (s.state, a.action, r, t.state, terminal) )
                batch_size -= 1
                s = t
        return batch
