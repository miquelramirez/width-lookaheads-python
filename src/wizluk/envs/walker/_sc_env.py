# -*- coding: utf-8 -*-

import copy
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
try:
    from gym.envs.classic_control import rendering
except Exception:
    print("Graphics not available!")
from . _sc_model import WalkBotSC

class WalkBotSCEnv(gym.Env):

    metadata = {
        'render.modes' : ['human','rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs):
        self.model = WalkBotSC(0, 'walker_0')

        # @TODO: allow task definition to be loaded from a file
        self.model.add_bounds('x', 0.0,10.0)
        self.model.add_bounds('y', 0.0,10.0)
        self.model.add_bounds('vx', -10.0,10.0)
        self.model.add_bounds('vy', -10.0,10.0)

        # control constants
        self.model.k_acc = 0.5
        self.model.k_dec = 2.0
        self.model.k_steer = 1.0
        self.R = np.zeros(5)
        self.R[1] = self.R[2] = np.power(self.model.k_steer,2)
        self.R[2] = np.power(self.model.k_acc,2)
        self.R[4] = np.power(self.model.k_dec,2)

        self.x0 = np.array( [2.0, 8.0, -1.0140,0.2664,0])
        self._sync_model(self.x0)

        self.x_G = np.matrix( [ [8.0], [2.0], [0.0], [0.0], [0]])
        tolerance = np.matrix( [ [0.25], [0.25], [0.2], [0.2],[0] ])
        self.G = (self.x_G - tolerance, self.x_G + tolerance)

        self.action_space = spaces.Discrete(len(self.model.supervisory_control_modes))
        self._build_obs()

        self.viewer = None
        self._s = None
        self._trajectory = [] # for rendering
        self.k = 0
        self.N = kwargs.get('max_episode_steps',100)

        self.done_on_invalid = kwargs.get('done_on_invalid', True)
        cost_function = kwargs.get('cost_function', 'shortest_path')
        if cost_function == 'shortest_path':
            self.reward = self.evaluate_shortest_path_cost
        else :
            self.Q = np.eye(5)
            self.Q[4,4] = 0.0
            self.reward = self.evaluate_QR_cost

        self.random_initial_state = kwargs.get('random_initial_state', False)

        self.perturb_velocities = kwargs.get('perturb_velocities', False)
        self.sigma_vx = float(kwargs.get('sigma_vx', 0.05))
        self.sigma_vy = float(kwargs.get('sigma_vy', 0.05))
        # We construct the \mu and \sigma for the state perturbation w_t
        self.mu = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        self.sigma = np.array([[0.0], [0.0], [self.sigma_vx], [self.sigma_vy], [0.0]])

    def _sync_model(self, s):
        self.model.x = s[0]
        self.model.y = s[1]
        self.model.vx = s[2]
        self.model.vy = s[3]
        self.model._control_mode = self.model._mode[int(s[4])]

    def _is_goal(self, s):
        s = np.squeeze(np.asarray(s))
        return np.all(self.G[0] <= s) and np.all(s <= self.G[1])

    def _is_valid(self,s):
        L = self.model.lower_bounds
        U = self.model.upper_bounds
        # Column vector into array
        s = np.squeeze(np.asarray(s))
        v_vec = np.array([s[2],s[3]])
        return np.all(L <= s) and np.all( s <= U) and np.linalg.norm(v_vec) > 1e-2

    def _build_obs(self):
        lo = self.model.lower_bounds
        hi = self.model.upper_bounds
        self.observation_space = spaces.Box(lo,hi, dtype=np.float32)
        assert self.observation_space.shape[0] == 5

    def seed( self, seed=None ) :
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def state(self) :
        return self._s

    @state.setter
    def state(self, v) :
        self._s = copy.deepcopy(v)

    def evaluate_shortest_path_cost(self, x):
        if self._is_goal(x): return 0.0
        return -1.0

    def evaluate_QR_cost(self, s):
        x = np.array([ [xi] for xi in s])
        xp = x - self.x_G
        return -(np.dot(xp.T,self.Q.dot(xp))[0,0]+self.R[int(s[-1])])


    def step(self, action) :
        self.k += 1
        if not self.done_on_invalid :
            if not self._is_valid(self.state):
                #print("INVALID: k: {}, N: {}, done: {}".format(self.k,self.N,self.k>=self.N))
                return self.state, self.reward(self.state), self.k >= self.N, {}

        # apply action
        #print(self.state)
        #print(action)
        self.state[-1] = action
        self._sync_model(self.state)
        next_state = self.model.simulate_evolution(dt=0.1)
        if self.perturb_velocities:
            # add noise perturbation
            next_state += self.sigma * np.random.randn(1,next_state.shape[1]) + self.mu
        next_state[-1] = action
        #print('Output of simulation: {}'.format(next_state))
        self.state = np.squeeze(np.asarray(next_state))
        #print(self.state)

        reward = self.reward(self.state)

        if self._is_goal(self.state):
            done = True
            #print("GOAL: k: {}, N: {}, done: {}".format(self.k,self.N,self.k>=self.N))
        elif not self._is_valid(self.state):
            #print("invalid state: {} L: {} U: {}".format(self.state,self.model.lower_bounds,self.model.upper_bounds))
            if self.done_on_invalid:
                done = True
                reward = -10000.0
            else :
                done = self.k >= self.N
            #print("INVALID: k: {}, N: {}, done: {}".format(self.k,self.N,done))
        else :
            done = self.k >= self.N

        #print("CONTINUE: k: {}, N: {}, done: {}".format(self.k,self.N,done))
        return self.state, reward, done, {}

    def reset(self, s = None) :
        if not self.random_initial_state:
            if s is None :
                self._s = self.x0
            else :
                self._s = copy.deepcopy(s)
        else:
            while True :
                self._s = np.random.randn(5)
                self._s[0] = 2.5*self._s[0] + 5.0
                self._s[1] = 2.5*self._s[1] + 5.0
                self._s[2] = 1.0*self._s[0] + 0.05
                self._s[3] = 1.0*self._s[1] + 0.05
                if self._is_valid(self.state): break
            self._s[-1] = 0 # always straight
        self._trajectory = []
        self.k = 0
        return self._s

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'viewer':
                setattr(result,k,None) # set the viewer pointer to None
            else :
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = 10.0
        world_height = 10.0
        xscale = screen_width/world_width
        yscale = screen_height/world_height
        robotwidth=0.25
        robotheight=0.25

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xy = [(0.0,0.0),(0.0, 10.0), (10.0,10.0), (10.0,0.0)]
            xys = [ (x*xscale,y*yscale) for x,y in xy]
            self.surface = rendering.make_polyline(xys)
            self.surface.set_linewidth(4)
            self.viewer.add_geom(self.surface)

            l,r,t,b = -robotwidth/2, robotwidth/2, robotheight/2, -robotheight/2
            self.robot = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.robot.set_color(0.0,0.0,0.8)
            self.robot.add_attr(rendering.Transform(translation=(0, 0)))
            self.robot_trans = rendering.Transform()
            self.robot.add_attr(self.robot_trans)
            self.viewer.add_geom(self.robot)

            l = self.G[0][0,0]*xscale
            b = self.G[0][1,0]*yscale
            r = self.G[1][0,0]*xscale
            t = self.G[1][1,0]*yscale
            self.goal_area = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.goal_area.set_color(0.8,0.0,0.0)
            self.viewer.add_geom(self.goal_area)


        self._trajectory.append( ( self.state[0]*xscale, self.state[1]*yscale) )
        if len(self._trajectory) >= 2 :
            move = rendering.Line(start=self._trajectory[-2],end=self._trajectory[-1])
            # orange: rgb(244, 215, 66)
            move.set_color(244.0/255.0,215.0/255.0,66.0/255.0)
            move.add_attr(rendering.LineStyle(0xAAAA))
            move.add_attr(rendering.LineWidth(4))
            self.viewer.add_geom(move)

        self.robot_trans.set_translation(self.state[0]*xscale, self.state[1]*yscale)
        self.robot_trans.set_rotation(np.arctan2(self.state[3],self.state[2]))
        self.robot_trans.set_scale(xscale,yscale)



        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
