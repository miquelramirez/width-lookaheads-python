import math
import copy
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class GridWorldEnv(gym.Env) :
    metadata = {
        'render.modes' : ['human','rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs) :
        self.action_space = spaces.Discrete(4)
        d = float(kwargs.get('dimension',4))
        self.observation_space = spaces.Box(low=0.0, high=d - 1.0, shape=(2,), dtype=np.float32)
        self.terminal_def = kwargs.get('terminal_def',"corners")
        self.use_obstacles = kwargs.get('obstacles',"False")
        if self.use_obstacles == "True":
            self.terminal_def = "center" # Obstacles only work with terminal as center
        self.initState = int(kwargs.get('initState', 0))

        initStates = [[0,0], [0, 1], [d/2 - 1, 0], [d/2 - 1, d/2 - 1], [d/2, d/2 - 2], [0,d/2], [d - 1, d - 1], [d - 1 - d/2 , d - 1], [d - 1 , d - 1 - d/2 ], [d - 2 , d - 1 - d/2 + 2]]
        assert(self.initState < len(initStates))
        self.initialState = initStates[self.initState]
        self.seed()
        self.viewer = None
        self._s = None
        self._trajectory = [] # for rendering
        self.steps_beyond_done = None
        self.action_effects = [ (1.0,0.0), (0.0,1.0), (-1.0,0.0), (0.0,-1.0) ]
        if self.terminal_def == "center" :
            self.terminals = [ np.array([d/2, d/2])]
        elif self.terminal_def == "corners" or self.terminal_def == "moving_diag" :
            #self.corners needed for checking when to change diag direction
            self.corners = [ np.array([ self.observation_space.low[0], self.observation_space.high[1]]), \
                np.array([self.observation_space.high[0], self.observation_space.low[1] ]) ]
            self.terminals = [ np.array([ self.observation_space.low[0], self.observation_space.high[1]]), \
                np.array([self.observation_space.high[0], self.observation_space.low[1] ]) ]
            self.diagDirectionx = [1, -1]
            self.diagDirectiony = [-1, 1]
        elif self.terminal_def == "boundary" :
            self.terminals = []
            for i in range(int(self.observation_space.high[0])):
                self.terminals.append(np.array([self.observation_space.high[0], i ]))
                self.terminals.append(np.array([i, self.observation_space.high[0]]))
            self.terminals.append(np.array([self.observation_space.high[0], self.observation_space.high[0]]))
        else :
            #The terminal definiton given for the GridWorld is invalid
            assert(False)

        self.obstacles = []
        if self.use_obstacles == "True":
            assert(self.terminal_def == "center")
            self.obstacles = [np.array([d/2 - 1, d/2]), np.array([d/2, d/2 - 1])] # Make grid below and left of goal a obstacle

            if d/2 + 1 < d - 1:
                self.obstacles.append(np.array([d/2 - 1, d/2 + 1]))
            if d/2 + 2 < d - 1:
                self.obstacles.append(np.array([d/2 - 1, d/2 + 2]))
            if d/2 + 3 < d - 1:
                self.obstacles.append(np.array([d/2, d/2 + 3]))

                addObstacle = 1
                while d/2 + addObstacle < d - 1 and addObstacle < 4:
                    self.obstacles.append(np.array([d/2 + addObstacle, d/2 + 3]))
                    self.obstacles.append(np.array([d/2 + addObstacle, d/2 - 1]))
                    addObstacle += 1

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
        dx, dy = self.action_effects[action]

        next_x = x + dx
        next_y = y + dy

        # Wraparound
        if self.observation_space.low[0] > next_x or \
            self.observation_space.high[0] < next_x or \
            self.observation_space.low[1] > next_y or \
            self.observation_space.high[1] < next_y :
            next_x = x
            next_y = y

        # Check for obstacles - results in no change to position
        pos = np.array([next_x, next_y])
        if any((pos == x).all() for x in self.obstacles):
            next_x = x
            next_y = y

        self.state = np.array([next_x, next_y])

        reward = -1.0
        done = False

        if self.terminal_def == "moving_diag" :
            self.terminals = [ np.array([ self.terminals[0][0] + self.diagDirectionx[0], self.terminals[0][1] + self.diagDirectiony[0]]), \
                np.array([self.terminals[1][0] + self.diagDirectionx[1], self.terminals[1][1] + self.diagDirectiony[1]]) ]
            for i in range(2) :
                if np.array_equal(self.terminals[i], self.corners[0]) :
                    self.diagDirectionx[i] = 1
                    self.diagDirectiony[i] = -1
                elif np.array_equal(self.terminals[i], self.corners[1]) :
                    self.diagDirectionx[i] = -1
                    self.diagDirectiony[i] = 1

        for t in self.terminals :
            if np.array_equal(self.state,t) :
                reward = 0.0
                done = True

        return self.state, reward, done, {}

    def reset(self, s = None) :
        if s is None :
            self._s = np.array(self.initialState)
        else :
            self._s = copy.deepcopy(s)
        self.steps_beyond_done = None
        if self.viewer: self.viewer.close()
        self.viewer = None
        self._trajectory = [] # for rendering
        if self.terminal_def == "moving_diag" :
            self.terminals = [ np.array([ self.observation_space.low[0], self.observation_space.high[1]]), \
                np.array([self.observation_space.high[0], self.observation_space.low[1] ]) ]
            self.diagDirectionx = [1, -1]
            self.diagDirectiony = [-1, 1]
        return self._s

    def getAdmissibleHeuristic(self):
        assert(self.terminal_def == "center") # Set up for use only when terminal in middle.
        #Manhattan Distance
        return np.min((- abs(self._s[0] - self.terminals[0][0]) - abs(self._s[1] - self.terminals[0][1]) + 1, 0))

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.observation_space.high[0] + 2
        world_height = self.observation_space.high[0] + 2
        xscale = screen_width/world_width
        yscale = screen_height/world_height
        robotwidth = 0.5
        robotheight = 0.5
        goalwidth = 0.5
        goalheight = 0.5

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #Grid
            for heightX in range(int(self.observation_space.high[0] + 1)):
                heightX = float(heightX)
                for widthX in range(int(self.observation_space.high[0] + 1)):
                    widthX = float(widthX)
                    xy = [(0.5 + widthX,0.5 + heightX),(0.5 + widthX, self.observation_space.high[0] + 1.5 - heightX), (self.observation_space.high[0] + 1.5- widthX,self.observation_space.high[0] + 1.5- heightX), (self.observation_space.high[0]- widthX + 1.5,0.5  + heightX)]
                    xys = [ (x*xscale,y*yscale) for x,y in xy]
                    self.surface = rendering.make_polyline(xys)
                    self.surface.set_linewidth(1)
                    self.viewer.add_geom(self.surface)

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

            self.obstacle_area = []
            self.obstacle_trans = []
            for k in range(len(self.obstacles)) :
                l,r,t,b = -goalwidth, goalwidth, goalheight, -goalheight
                self.obstacle_area.append(rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]))
                self.obstacle_area[k].set_color(0.0,0.8,0.0)
                self.obstacle_area[k].add_attr(rendering.Transform(translation=(0, 0)))
                self.obstacle_trans.append(rendering.Transform())
                self.obstacle_area[k].add_attr(self.obstacle_trans[k])
                self.viewer.add_geom(self.obstacle_area[k])

            self.goal_area = []
            self.goal_trans = []
            for k in range(len(self.terminals)) :
                l,r,t,b = -goalwidth, goalwidth, goalheight, -goalheight
                self.goal_area.append(rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]))
                self.goal_area[k].set_color(0.8,0.0,0.0)
                self.goal_area[k].add_attr(rendering.Transform(translation=(0, 0)))
                self.goal_trans.append(rendering.Transform())
                self.goal_area[k].add_attr(self.goal_trans[k])
                self.viewer.add_geom(self.goal_area[k])

            l,r,t,b = -robotwidth, robotwidth, robotheight, -robotheight
            self.robot = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.robot.set_color(0.0,0.0,0.8)
            self.robot.add_attr(rendering.Transform(translation=(0, 0)))
            self.robot_trans = rendering.Transform()
            self.robot.add_attr(self.robot_trans)
            self.viewer.add_geom(self.robot)

        self._trajectory.append( ( (self.state[0] + 1 ) * xscale, (self.state[1] + 1) * yscale) )
        for k in range(len(self.obstacles)) :
            self.obstacle_trans[k].set_translation((self.obstacles[k][0] + 1) * xscale, (self.obstacles[k][1] + 1) * yscale)
            self.obstacle_trans[k].set_scale(xscale,yscale)

        for k in range(len(self.terminals)) :
            self.goal_trans[k].set_translation((self.terminals[k][0] + 1) * xscale, (self.terminals[k][1] + 1) * yscale)
            self.goal_trans[k].set_scale(xscale,yscale)

        if len(self._trajectory) >= 2 :
            move = rendering.Line(start=self._trajectory[-2],end=self._trajectory[-1])
            # orange: rgb(244, 215, 66)
            move.set_color(244.0/255.0,215.0/255.0,66.0/255.0)
            move.add_attr(rendering.LineStyle(0xAAAA))
            move.add_attr(rendering.LineWidth(4))
            self.viewer.add_geom(move)

        self.robot_trans.set_translation((self.state[0] + 1) * xscale, (self.state[1] + 1) * yscale)
        self.robot_trans.set_scale(xscale,yscale)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer: self.viewer.close()
