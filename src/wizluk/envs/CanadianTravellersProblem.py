import math
import copy
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import random

# In this version of gridWorld we have a percentage chance of each grid cell,
# being blocked by some obstacle. The observable state is the entire cell, in which,
# a cell value of 0: has not been explored therefor it is unkown if it is blocked or not,
# a cell value of 1: if an explored cell which is free, a cell value of 2: is blocked,
# a value of 4: is the goal cell
class CTPEnv(gym.Env) :
    metadata = {
        'render.modes' : ['human','rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs) :
        self.action_space = spaces.Discrete(4)
        self.dimension = int(kwargs.get('dimension',4))
        d = self.dimension
        self.initPos = int(kwargs.get('initState', 0))
        initPositions = [[0,0], [0, 1], [d/2 - 1, 0], [d/2 - 1, d/2 - 1], [d/2, d/2 - 2], [0,d/2], [d - 1, d - 1], [d - 1 - d/2 , d - 1], [d - 1 , d - 1 - d/2 ], [d - 2 , d - 1 - d/2 + 2]]
        assert(self.initPos < len(initPositions))
        self.initPos = initPositions[self.initPos]
        self.observation_space = spaces.Box(low=0, high=4, shape=(int(self.dimension*self.dimension),), dtype=np.int)
        self.terminal_def = kwargs.get('terminal_def',"corners")
        self.use_obstacles = kwargs.get('obstacles',"False")
        self.seed()
        self.viewer = None
        self._s = None
        self._position = None
        self._trajectory = [] # for rendering

        self.action_effects = [ (1,0), (0,1), (-1,0), (0,-1) ]
        if self.terminal_def == "center" :
            self.terminals = [ np.array([d/2, d/2])]
        elif self.terminal_def == "corners" or self.terminal_def == "moving_diag" :
            #self.corners needed for checking when to change diag direction
            self.corners = [ np.array([ 0, self.dimension - 1 ]), \
                np.array([self.dimension - 1 , 0 ]) ]
            self.terminals = [ np.array([ 0, self.dimension - 1 ]), \
                np.array([self.dimension - 1 , 0 ]) ]
            self.diagDirectionx = [1, -1]
            self.diagDirectiony = [-1, 1]
        elif self.terminal_def == "boundary" :
            self.terminals = []
            for i in range(int(self.dimension - 1 )):
                self.terminals.append(np.array([self.dimension - 1 , i ]))
                self.terminals.append(np.array([i, self.dimension - 1 ]))
            self.terminals.append(np.array([self.dimension - 1 , self.dimension - 1 ]))
        else :
            #The terminal definiton given for the GridWorld is invalid
            assert(False)
        self.obstacles = []
        if self.use_obstacles == "True":
            assert(self.terminal_def == "center")
            self.possibleObstacles = [(np.array([d/2 - 1, d/2]),0.99), (np.array([d/2, d/2 - 1]), 0.99)] # Make grid below and left of goal a obstacle

            if d/2 + 1 < d - 1:
                self.possibleObstacles.append((np.array([d/2 - 1, d/2 + 1]),0.9))
            if d/2 + 2 < d - 1:
                self.possibleObstacles.append((np.array([d/2 - 1, d/2 + 2]), 0.9))
            if d/2 + 3 < d - 1:
                self.possibleObstacles.append((np.array([d/2, d/2 + 3]), 0.8))

                addObstacle = 1
                while d/2 + addObstacle < d - 1 and addObstacle < 10:
                    self.possibleObstacles.append((np.array([d/2 + addObstacle, d/2 + 3]), 0.4 - np.min((abs(0.4 / (10 - addObstacle)), 0.4))))
                    self.possibleObstacles.append((np.array([d/2 + addObstacle, d/2 - 1]), 0.4 - np.min((abs(0.4 / (10 - addObstacle)), 0.4))))
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
        x,y = self._position
        dx, dy = self.action_effects[action]

        next_x = x + dx
        next_y = y + dy

        # Wraparound
        if 0 > next_x or \
            self.dimension - 1  < next_x or \
            0 > next_y or \
            self.dimension - 1  < next_y :
            next_x = x
            next_y = y

        # Check for obstacles - results in no change to position

        if self._s[int(next_x * self.dimension + next_y)] == 2:
            next_x = x
            next_y = y
        elif self._s[int(next_x * self.dimension + next_y)] == 0:
            assert(False) #Should not have an unkown cell next to

        self._position = np.array([next_x, next_y])
        self.positionUpdate()

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
            if np.array_equal(self._position,t) :
                reward = 0.0
                done = True

        return self._s, reward, done, {}

    def checkPositionAndUpdate(self, ind):
        ind = int(ind)
        if self._s[ind] == 0:
            self._s[ind] = 1 # Assume empty
            for x, prob in self.possibleObstacles:
                if x[0] * self.dimension + x[1] == ind:
                    if random.random() < prob:
                        self._s[ind] = 2 #blocked
                        self.obstacles.append(np.array([np.floor(ind / (self.dimension)), int(ind % self.dimension)]))
                    break

    def positionUpdate(self):
        for pos in range(len(self.observation_space.low)):
            if self._s[pos] == 3:
                self._s[pos] = 1 # If agent was in cell must be unblocked
        self._s[int(self._position[0] * self.dimension + self._position[1])] = 3 #Set new position
        for ter in self.terminals:
            self._s[int(ter[0] * self.dimension + ter[1])] = 4 #Set new position
        #Update adjacent cells for partially obs version
        for action in range(4):
            dx, dy = self.action_effects[action]

            next_x = self._position[0] + dx
            next_y = self._position[1] + dy

            # check if valid index
            if 0 > next_x or \
                self.dimension - 1  < next_x or \
                0 > next_y or \
                self.dimension - 1  < next_y :
                continue
            else:
                self.checkPositionAndUpdate(next_x * self.dimension + next_y)

    def reset(self, s = None) :
        self.obstacles = []
        if s is None :
            self._s = np.zeros(len(self.observation_space.low))
            self._position = self.initPos
            self.positionUpdate()
        else :
            self._s = copy.deepcopy(s)

        if self.viewer: self.viewer.close()
        self.viewer = None
        self._trajectory = [] # for rendering
        if self.terminal_def == "moving_diag" :
            self.terminals = [ np.array([ 0, self.dimension - 1 ]), \
                np.array([self.dimension - 1 , 0 ]) ]
            self.diagDirectionx = [1, -1]
            self.diagDirectiony = [-1, 1]
        return self._s

    def getAdmissibleHeuristic(self):
        assert(self.terminal_def == "center") # Set up for use only when terminal in middle.
        #Manhattan Distance
        return np.min((- abs(self._position[0] - self.terminals[0][0]) - abs(self._position[1] - self.terminals[0][1]) + 1, 0))

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.dimension - 1  + 2
        world_height = self.dimension - 1  + 2
        xscale = screen_width/world_width
        yscale = screen_height/world_height
        robotwidth = 0.5
        robotheight = 0.5
        goalwidth = 0.5
        goalheight = 0.5

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #Grid
            for heightX in range(int(self.dimension - 1  + 1)):
                heightX = float(heightX)
                for widthX in range(int(self.dimension - 1  + 1)):
                    widthX = float(widthX)
                    xy = [(0.5 + widthX,0.5 + heightX),(0.5 + widthX, self.dimension - 1  + 1.5 - heightX), (self.dimension - 1  + 1.5- widthX,self.dimension - 1  + 1.5- heightX), (self.dimension - 1 - widthX + 1.5,0.5  + heightX)]
                    xys = [ (x*xscale,y*yscale) for x,y in xy]
                    self.surface = rendering.make_polyline(xys)
                    self.surface.set_linewidth(1)
                    self.viewer.add_geom(self.surface)

            xy = [(0.5,0.5),(0.5, self.dimension - 1  + 1.5), (self.dimension - 1  + 1.5,self.dimension - 1  + 1.5), (self.dimension - 1  + 1.5,0.5)]
            xys = [ (x*xscale,y*yscale) for x,y in xy]
            self.surface = rendering.make_polyline(xys)
            self.surface.set_linewidth(4)
            self.viewer.add_geom(self.surface)

            xy = [(0.5,0.5),(self.dimension - 1  + 1.5,0.5)]
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


        self._trajectory.append( ( (self._position[0] + 1 ) * xscale, (self._position[1] + 1) * yscale) )
        for k in range(len(self.obstacles)) :
            try:
                self.obstacle_trans[k]
            except:
                l,r,t,b = -goalwidth, goalwidth, goalheight, -goalheight
                self.obstacle_area.append(rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]))
                self.obstacle_area[k].set_color(0.0,0.8,0.0)
                self.obstacle_area[k].add_attr(rendering.Transform(translation=(0, 0)))
                self.obstacle_trans.append(rendering.Transform())
                self.obstacle_area[k].add_attr(self.obstacle_trans[k])
                self.viewer.add_geom(self.obstacle_area[k])

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

        self.robot_trans.set_translation((self._position[0] + 1) * xscale, (self._position[1] + 1) * yscale)
        self.robot_trans.set_scale(xscale,yscale)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer: self.viewer.close()
