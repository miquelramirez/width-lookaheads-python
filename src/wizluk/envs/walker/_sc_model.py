# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as constants
from scipy.integrate import ode
import math

class WalkBotSC(object) :

    def __init__(self, id, name) :
        self._id = id
        self._name = name
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._k_steer = 0.0
        self._k_dec = 0.0
        self._k_acc = 0.0
        self._bounds = { 'x' : [-float('inf'),float('inf')], 'y' : [-float('inf'),float('inf')], \
                        'vx' : [-float('inf'),float('inf')], 'vy' : [-float('inf'),float('inf')],\
                        'mode' : [0,4]}
        # control
        self._control_mode = 'straight'
        self._mode = [ 'straight', 'left', 'right', 'accelerating', 'decelerating']
        self._mode_index = { name : k for k,name in enumerate(self._mode) }
        self._mode_dynamics = {
            'straight' : lambda t, x : self.straight_dynamics(x),
            'left' : lambda t, x : self.left_dynamics(x),
            'right' : lambda t, x : self.right_dynamics(x),
            'accelerating': lambda t, x : self.accelerating_dynamics(x),
            'decelerating': lambda t, x : self.decelerating_dynamics(x)
        }

    @property
    def supervisory_control_modes(self):
        return self._mode


    def add_bounds(self, var, lb, ub ) :
        try :
            self._bounds[var]= [lb,ub]
        except KeyError :
            print("Var '{}' does not exist".format(var))

    @property
    def lower_bounds(self) :
        return np.array([self._bounds[var][0] for var in ['x','y','vx','vy', 'mode']])

    @property
    def upper_bounds(self) :
        return np.array([self._bounds[var][1] for var in ['x','y','vx','vy', 'mode']])

    @property
    def id(self) :
        return self._id
    @property
    def name(self) :
        return self._name

    @property
    def k_steer(self) :
        return self._k_steer
    @k_steer.setter
    def k_steer(self, v) :
        self._k_steer = v

    @property
    def k_acc(self) :
        return self._k_acc
    @k_acc.setter
    def k_acc(self, v) :
        self._k_acc = v

    @property
    def k_dec(self) :
        return self._k_dec
    @k_dec.setter
    def k_dec(self, v) :
        self._k_dec = v

    @property
    def x(self) :
        return self._x
    @x.setter
    def x(self, value) :
        self._x = value
    @property
    def y(self) :
        return self._y
    @y.setter
    def y(self, value) :
        self._y = value

    @property
    def vx(self) :
        return self._vx
    @vx.setter
    def vx(self, value) :
        self._vx = value
    @property
    def vy(self) :
        return self._vy
    @vy.setter
    def vy(self, value) :
        self._vy = value

    @property
    def mode(self) :
        return self._control_mode
    @mode.setter
    def mode(self,v) :
        if type(v) == int :
            self._control_mode = self._mode[v]
            return

        assert v in self._mode_index.keys()
        self._control_mode = v

    @property
    def state(self) :
        return np.matrix( [[self._x], [self._y],\
                            [self._vx], [self._vy],\
                            [self._mode_index[self._control_mode]] ])
    @state.setter
    def state(self, s ) :
        assert type(s) == np.matrix
        self._x = s[0,0]
        self._y = s[1,0]
        self._vx = s[2,0]
        self._vy = s[3,0]
        self._control_mode = self._mode[int(s[4,0])]

    def __integrate( self, M, x0, h, t0 = 0.0 ) :
        r = ode(M).set_integrator('dopri5')
        r.set_initial_value(x0, t0)
        #y = np.matrix(r.integrate(r.t+h))
        y = r.integrate(r.t+h)
        if not r.successful() :
            raise RuntimeError("Integration failed!")
        return y

    def straight_dynamics( self, x ) :
        y = np.array( [\
                        [ x[2] ],\
                        [ x[3] ],\
                        [ 0.0 ],\
                        [ 0.0 ],\
                        [ 0.0 ]\
                        ])
        return y

    def left_dynamics( self, x ) :
        y = np.array( [\
                        [ x[2] ],\
                        [ x[3] ],\
                        [ -self._k_steer * x[3] ],\
                        [ self._k_steer * x[2] ],\
                        [ 0.0 ]\
                        ])
        return y

    def right_dynamics( self, x) :
        y = np.array( [\
                        [ x[2] ],\
                        [ x[3] ],\
                        [ self._k_steer * x[3] ],\
                        [ -self._k_steer * x[2] ],\
                        [ 0.0 ]\
                        ])
        return y

    def accelerating_dynamics( self, x ) :
        y = np.array( [\
                        [ x[2] ],\
                        [ x[3] ],\
                        [ self._k_acc * x[2] ],\
                        [ self._k_acc * x[3] ],\
                        [ 0.0 ]\
                        ])
        return y

    def decelerating_dynamics( self, x ) :
        y = np.array( [\
                        [ x[2] ],\
                        [ x[3] ],\
                        [ -self._k_dec * x[2] ],\
                        [ -self._k_dec * x[3] ],\
                        [ 0.0 ]\
                        ])
        return y


    def simulate_evolution( self, dt = 0.01 ) :
        x0 = self.state
        #print('{}_dynamics, dt={}'.format(self._control_mode,dt))
        return self.__integrate(self._mode_dynamics[self._control_mode], x0, dt)

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
