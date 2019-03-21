import math
import copy
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class CliffWorldEnv(gym.Env) :
    metadata = {
        'render.modes' : ['human','rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs) :
        self.action_space = spaces.Discrete(4)
        self.W = kwargs.get('width')
        self.H = kwargs.get('height')
        self.sticky_prob = kwargs.get('sticky_prob')
        lower_bounds = np.array([0.0, 0.0])
        upper_bounds = np.array([self.W-1.0, self.H-1.0])
        self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.float32)
        self.seed()
        self.viewer = None
        self._s = None
        self._trajectory = [] # for rendering
        self.action_effects = [ (1.0,0.0), (0.0,1.0), (-1.0,0.0), (0.0,-1.0) ]
        self.initial = np.array([0.0, 0.0])
        self.terminal = np.array([self.W - 1.0, 0.0])
        self.previous_action = None


    def seed( self, seed=None ) :
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def state(self) :
        return self._s

    @state.setter
    def state(self, v) :
        self._s = copy.deepcopy(v)

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

    def step(self, action) :
        x,y = self._s

        if self.sticky_prob > 0.0 :
            if np.random.rand() > self.sticky_prob:
                current_action = action
            else:
                if self.previous_action is not None:
                    current_action = self.previous_action
                else:
                    current_action = action
        else:
            current_action = action
        self.previous_action = action

        dx, dy = self.action_effects[current_action]
        next_x = x + dx
        next_y = y + dy

        # Wraparound
        if self.observation_space.low[0] > next_x or \
            self.observation_space.high[0] < next_x or \
            self.observation_space.low[1] > next_y or \
            self.observation_space.high[1] < next_y :
            next_x = x
            next_y = y

        self.state = np.array([next_x, next_y])


        # Terminal
        if np.array_equal(self.state, self.terminal):
            return self.state, -1.0, True, {}

        # Cliff
        if (self.state[0] >= 1.0) and (self.state[0] <= self.W-2.0) \
            and self.state[1] == 0.0:
            self.state = self.initial
            return self.state, -100.0, False, {}

        return self.state, -1.0, False, {}

    def reset(self) :
        self._s = self.initial
        if self.viewer: self.viewer.close()
        self.viewer = None
        self._trajectory = [] # for rendering

        return self._s

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.observation_space.high[0] + 2
        world_height = self.observation_space.high[1] + 2
        xscale = screen_width/world_width
        yscale = screen_height/world_height
        robotwidth = 0.5
        robotheight = 0.5
        goalwidth = 0.5
        goalheight = 0.5

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xy = [(0.5,0.5),(0.5, self.observation_space.high[0] + 1.5), (self.observation_space.high[0] + 1.5,self.observation_space.high[0] + 1.5), (self.observation_space.high[0] + 1.5,0.5)]
            xys = [ (x*xscale,y*yscale) for x,y in xy]
            self.surface = rendering.make_polyline(xys)
            self.surface.set_linewidth(4)
            self.viewer.add_geom(self.surface)

            xy = [(0.5,0.5),(self.observation_space.high[0] + 1.5,0.5)]
            xys = [ (x*xscale,y*yscale) for x,y in xy]
            self.surface = rendering.make_polyline(xys)
            self.surface.set_linewidth(4)
            self.viewer.add_geom(self.surface)

            l,r,t,b = -robotwidth, robotwidth, robotheight, -robotheight
            self.robot = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.robot.set_color(0.0,0.0,0.8)
            self.robot.add_attr(rendering.Transform(translation=(0, 0)))
            self.robot_trans = rendering.Transform()
            self.robot.add_attr(self.robot_trans)
            self.viewer.add_geom(self.robot)

            self.goal_area = None
            self.goal_trans = None

            l,r,t,b = -goalwidth, goalwidth, goalheight, -goalheight
            self.goal_area = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.goal_area.set_color(0.8,0.0,0.0)
            self.goal_area.add_attr(rendering.Transform(translation=(0, 0)))
            self.goal_trans = rendering.Transform()
            self.goal_area.add_attr(self.goal_trans)
            self.viewer.add_geom(self.goal_area)


        self._trajectory.append( ( (self.state[0] + 1 ) * xscale, (self.state[1] + 1) * yscale) )
        if len(self._trajectory) >= 2 :
            move = rendering.Line(start=self._trajectory[-2],end=self._trajectory[-1])
            # orange: rgb(244, 215, 66)
            move.set_color(244.0/255.0,215.0/255.0,66.0/255.0)
            move.add_attr(rendering.LineStyle(0xAAAA))
            move.add_attr(rendering.LineWidth(4))
            self.viewer.add_geom(move)

        self.robot_trans.set_translation((self.state[0] + 1) * xscale, (self.state[1] + 1) * yscale)
        self.robot_trans.set_scale(xscale,yscale)

        self.goal_trans.set_translation((self.terminal[0] + 1) * xscale, (self.terminal[1] + 1) * yscale)
        self.goal_trans.set_scale(xscale,yscale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer: self.viewer.close()
