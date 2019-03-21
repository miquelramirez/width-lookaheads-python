# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as constants
from scipy.integrate import ode
import math
from wizluk.errors import WizlukError
import scipy as sp
import scipy.signal as sig
from cvxpy import *

class WalkBot(object) :

    def __init__(self, id, name, dt=0.5) :
        self._id = id
        self._name = name

        self._state_matrix = np.array([[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0],np.zeros(4),np.zeros(4)])
        self._input_matrix = np.array([np.zeros(2), np.zeros(2), [1.0,0.0], [0.0,1.0]])

        self._state_bounds = { 'x' : [-float('inf'),float('inf')],
                        'y' : [-float('inf'),float('inf')],
                        'vx' : [-float('inf'),float('inf')],
                        'vy' : [-float('inf'),float('inf')]}
        self._input_bounds = {
                        'ax' : [-float('inf'),float('inf')],
                        'ay' : [-float('inf'),float('inf')]}
        self.h = []
        # discretise dynamics
        cont_sys = sig.lti( self._state_matrix, self._input_matrix, np.identity(self._state_matrix.shape[0]), np.zeros(self._input_matrix.shape))
        self.disc_sys = cont_sys.to_discrete(dt)


    def update_discretisation(self, dt):
        cont_sys = sig.lti( self._state_matrix, self._input_matrix, np.identity(self._state_matrix.shape[0]), np.zeros(self._input_matrix.shape))
        self.disc_sys = cont_sys.to_discrete(dt)


    def make_constraints(self):
        def bounds_constraint(lb, v, ub):
            return lb <= v <= ub
        return [ lambda x,u : bounds_constraint(self._state_bounds['x'][0], x[0], self._state_bounds['x'][1]),
                lambda x,u : bounds_constraint(self._state_bounds['y'][0], x[1], self._state_bounds['y'][1]),
                lambda x,u : bounds_constraint(self._state_bounds['vx'][0], x[2], self._state_bounds['vx'][1]),
                lambda x,u : bounds_constraint(self._state_bounds['vy'][0], x[3], self._state_bounds['vy'][1]),
                lambda x,u : bounds_constraint(self._input_bounds['ax'][0], u[0], self._input_bounds['ax'][1]),
                lambda x,u : bounds_constraint(self._input_bounds['ay'][0], u[1], self._input_bounds['ay'][1])]

    def mpc(self, x0, xG, N, Q, R, dt=0.5, verbose=False):

        #x = Variable(self._state_matrix.shape[0], N+1)
        #u = Variable(self._input_matrix.shape[1], N)
        x = [Variable(self._state_matrix.shape[0])]
        u = []

        s_lb, u_lb = self.lower_bounds
        s_ub, u_ub = self.upper_bounds

        u_lb += 1e-1
        u_ub -= 1e-1

        constraints = [ x[0] == x0 ] # initial state
        cost = 0

        for t in range(N):
            ut = Variable(self._input_matrix.shape[1])
            x_next = Variable(self._state_matrix.shape[0])
            u += [ut]
            x += [x_next]

            constraints += [ x[t+1] == self.disc_sys.A*x[t] + self.disc_sys.B*u[t] ]
            constraints += [u[t] <= u_ub, u_lb <= u[t], x[t] <= s_ub,s_lb <= x[t] ]
            cost += quad_form(x[t] - xG, Q) + quad_form(u[t], R)

        objective = Minimize(cost)
        prob = Problem(objective, constraints)
        #print(prob)
        sol = prob.solve(verbose=verbose)
        #print(sol)
        xs = np.array(list(map( lambda var: var.value, x)))
        us = np.array(list(map( lambda var: var.value, u)))

        # We return the first control input
        return us[0]

    def add_bounds(self, var, lb, ub ) :
        if var in self._state_bounds:
            self._state_bounds[var]= [lb,ub]
            #print("Set bounds for var {} to {}".format(var,[lb,ub]))
        elif var in self._input_bounds:
            self._input_bounds[var] = [lb,ub]
            #print("Set bounds for var {} to {}".format(var,[lb,ub]))
        else:
            raise WizlukError("pnl.envs.walker.WalkBot.add_bounds(): Variable '{}' does not exist".format(var))
        self.h = self.make_constraints()
        #print(self.lower_bounds)
        #print(self.upper_bounds)

    @property
    def lower_bounds(self) :
        return np.array([self._state_bounds[var][0] for var in ['x','y','vx','vy']]),\
                np.array([self._input_bounds[var][0] for var in ['ax','ay']])

    @property
    def upper_bounds(self) :
        return np.array([self._state_bounds[var][1] for var in ['x','y','vx','vy']]),\
                np.array([self._input_bounds[var][1] for var in ['ax','ay']])

    @property
    def id(self) :
        return self._id
    @property
    def name(self) :
        return self._name

    def __integrate( self, M, x0, h, t0 = 0.0 ) :
        r = ode(M).set_integrator('dopri5')
        r.set_initial_value(x0, t0)
        y = r.integrate(r.t+h)
        if not r.successful() :
            raise RuntimeError("Integration failed!")
        return y

    def simulate_evolution( self, x0, u, dt = 0.01 ) :
        try:
            u = np.reshape(u,(self._input_matrix.shape[1],1))
        except ValueError as e:
            raise RuntimeError("u: {}".format(u))
        def f(t,x):
            x = np.reshape(x,(self._state_matrix.shape[0],1))
            delta = self._state_matrix.dot(x) + self._input_matrix.dot(u)
            #print(delta)
            return delta
        return self.__integrate(f, x0, dt)

    @staticmethod
    def draw_trajectory( ax, trajectory, **kwargs ) :
        from matplotlib.patches import Arrow

        h = kwargs.get('dt', 0.01)


        t_values = np.linspace( 0, len(trajectory)*h, num=len(trajectory)+1)
        x_values = [ x[0,0] for x in trajectory ]
        y_values = [ x[1,0] for x in trajectory]

        ax.plot( x_values, y_values, \
                color=kwargs.get('color','b'), \
                linestyle=kwargs.get('linestyle', '-'), \
                markersize = 10)

        v_vec = Arrow( x_values[0], y_values[0], x_values[1], y_values[1], width=0.5, color='g', alpha=0.4 )
        ax.add_patch(v_vec) # this draws the velocity vector
