# -*- coding: utf-8 -*-

import copy
import random
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
try:
    from gym.envs.classic_control import rendering
except Exception:
    print("Graphics not available!")
from . _model import WalkBot

class WalkBotEnv(gym.Env):

    metadata = {
        'render.modes' : ['human','rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs):
        self.model = WalkBot(0, 'walker_0')

        # @TODO: allow task definition to be loaded from a file
        self.model.add_bounds('x', 0.0,10.0)
        self.model.add_bounds('y', 0.0,10.0)
        self.model.add_bounds('vx', -10.0,10.0)
        self.model.add_bounds('vy', -10.0,10.0)
        self.model.add_bounds('ax', -2.0,2.0)
        self.model.add_bounds('ay', -2.0,2.0)
        self.t = 0


        self.x0 = np.array( [[2.0], [8.0], [-1.0140],[0.2664]])
        self.x_G = np.matrix( [ [8.0], [2.0], [0.0], [0.0]])
        tolerance = np.matrix( [ [0.25], [0.25], [0.2], [0.2] ])
        self.G = (self.x_G - tolerance, self.x_G + tolerance)

        self._build_actions()
        self._build_obs()

        self.viewer = None
        self._s = None
        self._trajectory = [] # for rendering
        cost_function = kwargs.get('cost_function', 'shortest_path')
        if cost_function == 'shortest_path':
            self.reward = self.evaluate_shortest_path_cost
            self.reward_goal = self.evaluate_shortest_path_cost
        else :
            self.Q = np.eye(4)
            self.R = np.eye(2)
            self.reward = self.evaluate_QR_cost

        self.random_initial_state = kwargs.get('random_initial_state', False)

        self.perturb_velocities = kwargs.get('perturb_velocities', False)
        self.sigma_vx = float(kwargs.get('sigma_vx', 0.05))
        self.sigma_vy = float(kwargs.get('sigma_vy', 0.05))
        # We construct the \mu and \sigma for the state perturbation w_t
        self.mu = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.sigma = np.array([[0.0], [0.0], [self.sigma_vx], [self.sigma_vy]])
        self.horizon = 100


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


    def _sync_model(self, s):
        self.model.x = s[0]
        self.model.y = s[1]
        self.model.vx = s[2]
        self.model.vy = s[3]
        self.model._control_mode = self.model._mode[int(s[4])]

    def _is_goal(self, x):
        s = np.squeeze(np.asarray(x))
        return np.all(self.G[0] <= s) and np.all(s <= self.G[1])

    def _is_valid(self,x,u):
        return all(h(x.flatten(),u.flatten()) for h in self.model.h) and (np.linalg.norm(x[2:,0]) >= 0.01)
        #Lx, Lu = self.model.lower_bounds
        #Ux, Uu = self.model.upper_bounds
        # Column vector into array
        #s = np.squeeze(np.asarray(x))
        #a = np.squeeze(np.asarray(u))
        #return np.all(Lx <= s) and np.all( s <= Ux) and np.all(Lu <= a) and np.all(a <= Uu)

    def _build_obs(self):
        lo, _ = self.model.lower_bounds
        hi, _ = self.model.upper_bounds
        self.observation_space = spaces.Box(lo,hi, dtype=np.float32)
        assert self.observation_space.shape[0] == 4

    def _build_actions(self):
        _, lo = self.model.lower_bounds
        _, hi = self.model.upper_bounds
        self.action_space = spaces.Box(lo,hi, dtype=np.float32)
        assert self.action_space.shape[0] == 2


    def seed( self, seed) :

        if type(seed) == int:
            random.seed(seed)
            np.random.seed(seed)
            while True :
                sigma_x0 = np.array([[2.5],[2.5],[1.0],[1.0]])
                mu_x0 = np.array([[5.0],[5.0],[0.05],[0.05]])
                self.rnd_x0 = np.multiply(sigma_x0,np.random.randn(4,1))+mu_x0
                dummy_action = np.zeros((2,1))
                if self._is_valid(self.rnd_x0,dummy_action): break
        else:
            self._s = seed
        return [seed]

    @property
    def state(self) :
        return self._s

    @state.setter
    def state(self, v) :
        self._s = copy.deepcopy(v)

    def evaluate_shortest_path_cost(self, x,u):
        if self._is_goal(x): return 0.0
        return -1.0

    def evaluate_QR_cost(self, x, u):
        x = np.reshape(x,(self.Q.shape[0],1))
        u = np.reshape(u,(self.R.shape[1],1))
        xp = x - self.x_G
        J = np.dot(xp.T,self.Q.dot(xp)) + np.dot(u.T,self.R.dot(u))
        return -J[0,0]

    def step(self, action) :
        info = {}
        valid = self._is_valid(self.state,action)
        if valid:
            # apply action
            #print(self.state)
            #print(action)
            self.state = self.model.simulate_evolution(self.state,action,dt=0.1)
            #print(self.state)
            if self.perturb_velocities:
                #print(self.sigma.shape)
                w_t = np.multiply(self.sigma, np.random.randn(4,1)) + self.mu
                #print(w_t)
                self.state = self.state + w_t
            #print(self.state)

        info['valid'] = valid
        #if self._is_goal(self.state):
        reward = self.reward(self.state,action)

        if self._is_goal(self.state):
            info['goal'] = True
        else:
            info['goal'] = False

        self.t += 1
        done = False
        if self.t >= self.horizon-1:
            # MRJ: g(xN) term
            reward += self.reward(self.state,np.zeros(2))
            done = True

        return self.state, reward, done, info

    def reset(self, s = None) :
        if not self.random_initial_state:
            if s is None :
                self._s = self.x0
            else :
                self._s = copy.deepcopy(s)
        else:
            self._s = copy.deepcopy(self.rnd_x0)
        self.t = 0
        self._trajectory = []
        assert self._s is not None
        return self._s

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
