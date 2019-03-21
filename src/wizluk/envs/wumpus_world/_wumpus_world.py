import math
import copy
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

#Adapted from https://github.com/SaricVr/gym-wumpus/blob/master/gym_wumpus/envs/wumpus_env.py

class WumpusWorldEnv(gym.Env) :
    metadata = {
        'render.modes' : ['human','rgb_array'],
        'video.frames_per_second' : 50
    }
    """
    The wumpus world was introduced by Genesereth, and is discussed in Russell-Norvig.
    The wumpus world is a cave consisting of rooms connected by passageways. Lurking somewhere in the cave is the
    terrible wumpus, a beast that eats anyone who enters its room. The wumpus can be shot by an agent, but the agent has
    only one arrow. Some rooms contain bottomless pits that will trap anyone who wanders into these rooms (except for
    the wumpus, which is too big to fall in). The only mitigating feature of this bleak environment is the
    of finding a heap of gold. The game ends either when the agent dies (falling into a pit or being eaten by the
    wumpus) or when the agent climbs out of the cave.
    Performance mesure:
    -   +1000 for climbing out of the cave with the gol
    -   -1000 for falling into a pit or being eaten by the wumpus
    -   -1 for each action taken
    -   -10 for using up the arrow
    Actions:
        - 'Forward' (F): moves the agent forward of one cell (walls are blocking)
        - 'TurnLeft' by 90 (L)
        - 'TurnRight' by 90 (R)
        - 'GrabGold' (G): pick up the gold if it is in the same cell as the agent
        - 'ShootArrow' (S): fires an arrow in a straight line in the direction of the agent is facing. The arrow
          continues until it either hits (and hece kills) the wumpus or hits a wall. The agent has only 1 arrow.
        - 'ClimbOut' (C): climbs out of the cave
    Perceptions:
        - In the square containing the wumpus (W) and in the directly (not diagonally) adjacent squares, the agent will
          perceive a 'Stench' (S)
        - In the squares directly (not diagonally) adjacent to a pit (P), the agent will perceive a 'Breeze' (B)
        - In the square where the gold (G) is, the agent wil perceive a 'Glitter'
        - When an agent walks into a wall, it will perceive a 'Bump'
        - When the wumpus is killed, it emits a woeful 'Scream' that can be perceived anywhere in the cave
    Perceptions are givent to the agent in the form of a tuple of booleans for
    ('Stench','Breeze','Glitter','Bump','Scream'). For example if there is a stench and a breeze, but no glitter,
    bump, or scream, the perception tuple will be: (True, True, False, False, False).
    Environment:
    A fixed 4x4 grid of rooms. The agent always starts in the square labeled [0,0], facing to the right.
    S     N     B   P
    W   B/S/G   P   B
    S     N     B   N
    A     B     P   B
    """

    def __init__(self, **kwargs) :
        self.oberservable = kwargs.get('oberservable',"full")
        self.action_space = spaces.Discrete(6)
        self.action_effects = ['F', 'L', 'R', 'G', 'S', 'C']
        self.setting = kwargs.get('setting',"classic") #'classic' or 'gold_on_shortest_path'
        self.deviationOfGoldFromShortestPath = kwargs.get('deviation_of_gold_from_shortest_path', 1) # can be 1, 5, 6, 7 or 8 for'gold_on_shortest_path' setting
        self.random_action_prob = kwargs.get('random_action_prob', -1)
        if self.oberservable == "full":
            lower_bounds = np.zeros((21)) * -1
            upper_bounds = np.ones((21)) * 3
            self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.int)
        elif self.oberservable == "perceptions":
            lower_bounds = np.array([0, 0 ,0 ,0 ,0])
            upper_bounds = np.array([1, 1, 1, 1, 1])
            self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.int)

        self.seed()
        self.viewer = None
        self._s = None


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
        x,y,orientation,arrow, hasGold = self._s[0]
        if self.random_action_prob > 0.0:
            if np.random.rand() < self.random_action_prob:
                action = np.random.randint(0, len(self.action_effects))
        action_ = self.action_effects[action]

        x = int(x)
        y = int(y)

        reward = -1.0
        done = False
        didBumpIntoWall = False
        didKillWumpus = False

        if action_ == 'F':
            if orientation == 0:
                if y == 3:
                    #Bump against wall
                    didBumpIntoWall = True
                else:
                    #Perform move and check for pit or wumpus for termination
                    self._s[0][1] += 1
                    done = self.checkForWumpusOrPit()
                    if done:
                        reward -= 1000
            if orientation == 1:
                if x == 3:
                    #Bump against wall
                    didBumpIntoWall = True
                else:
                    #Perform move and check for pit or wumpus for termination
                    self._s[0][0] += 1
                    done = self.checkForWumpusOrPit()
                    if done:
                        reward -= 1000

            if orientation == 2:
                if y == 0:
                    #Bump against wall
                    didBumpIntoWall = True
                else:
                    #Perform move and check for pit or wumpus for termination
                    self._s[0][1] -= 1
                    done = self.checkForWumpusOrPit()
                    if done:
                        reward -= 1000

            if orientation == 3:
                if x == 0:
                    #Bump against wall
                    didBumpIntoWall = True
                else:
                    #Perform move and check for pit or wumpus for termination
                    self._s[0][0] -= 1
                    done = self.checkForWumpusOrPit()
                    if done:
                        reward -= 1000

        elif action_ == 'L':
            if orientation == 0:
                self._s[0][2] = 3
            else:
                self._s[0][2] -= 1

        elif action_ == 'R':
            if orientation == 3:
                self._s[0][2] = 0
            else:
                self._s[0][2] += 1

        elif action_ == 'G':

            if self._s[1][x][y] == 3:
                # Grab gold
                self._s[0][4] = 1
                self._s[1][x][y] = 0

        elif action_ == 'C':
            if self.setting == 'gold_on_shortest_path':
                if self.deviationOfGoldFromShortestPath == 1:
                    leaveCavex = 1
                    leaveCavey = 2
                elif self.deviationOfGoldFromShortestPath == 5:
                    leaveCavex = 1
                    leaveCavey = 1
                elif self.deviationOfGoldFromShortestPath == 6:
                    leaveCavex = 2
                    leaveCavey = 1
                elif self.deviationOfGoldFromShortestPath == 7:
                    leaveCavex = 1
                    leaveCavey = 0
                elif self.deviationOfGoldFromShortestPath == 8:
                    leaveCavex = 0
                    leaveCavey = 1
                else:
                    assert(False)
            else:
                leaveCavex = 0 # Classic setting where exit is at start
                leaveCavey = 0

            if x == leaveCavex and y == leaveCavey:
                # Climb out of cave at start pos, terminal, reward of 1000 if gold has been picked up
                done = True
                self._s[0][0] = -1
                self._s[0][1] = -1


                if hasGold == 1:
                    reward += 1000

        elif action_ == 'S':
            if self._s[0][3] > 0:
                reward -= 10
                self._s[0][3] -= 1 # reduce number of arrows by one
                #check if wumpus is in line of arrow

                if orientation == 0:
                    checkY = y + 1
                    while checkY < 4:
                        if self._s[1][x][int(checkY)] == 2:
                            self._s[1][x][int(checkY)] = 0
                            didKillWumpus = True
                        checkY += 1

                if orientation == 1:
                    checkX = x + 1
                    while checkX < 4:
                        if self._s[1][int(checkX)][y] == 2:
                            self._s[1][int(checkX)][y] = 0
                            didKillWumpus = True
                        checkX += 1

                if orientation == 2:
                    checkY = y - 1
                    while checkY > -1:
                        if self._s[1][x][int(checkY)] == 2:
                            self._s[1][x][int(checkY)] = 0
                            didKillWumpus = True
                        checkY -= 1

                if orientation == 3:
                    checkX = x - 1
                    while checkX > -1:
                        if self._s[1][int(checkX)][y] == 2:
                            self._s[1][int(checkX)][y] = 0
                            didKillWumpus = True
                        checkX -= 1

        else:
            assert(False)

        if self.oberservable == "full":
            return np.array(np.append(np.reshape(self._s[1],[1,16]), np.reshape(self._s[0], [1, 5]))), reward, done, {}
        elif self.oberservable == "perceptions":
            return self.getPerceptions(didBumpIntoWall, didKillWumpus), reward, done, {}

    def checkForWumpusOrPit(self):
        if self._s[1][int(self._s[0][0])][int(self._s[0][1])] == 1 or self._s[1][int(self._s[0][0])][int(self._s[0][1])] == 2:
            return True
        return False

    def getPerceptions(self, didBumpIntoWall, didKillWumpus):
        perceptions = np.zeros(5)
        # ('Stench','Breeze','Glitter','Bump','Scream')
        for i in range(4):
            for j in range(4):
                if np.abs(i - self._s[0][0]) + np.abs(j - self._s[0][1]) == 1: #Agent is adjacent to this cell
                    if self._s[1][i][j] == 2: # Check if adjacent to wumpus
                        perceptions[0] = 1

                    if self._s[1][i][j] == 1: # Check if adjacent to pit
                        perceptions[1] = 1

        if self._s[1][int(self._s[0][0])][int(self._s[0][1])] == 3: # Check for gold at current position
            perceptions[2] = 1

        if didBumpIntoWall: # Check for bump
            perceptions[3] = 1

        if didKillWumpus: # Check for wumpus scream
            perceptions[4] = 1
        return perceptions

    def reset(self, s = None) :
        if s is None :
            self._s = [np.zeros(5), np.zeros((4,4))]
            self._s[0][0] = 0 #Agent x pos
            self._s[0][1] = 0 #Agent y pos
            self._s[0][2] = 1 #Agent orientation [0: up, 1: right, 2: down, 3:left]
            self._s[0][3] = 1 #Num of arrows left
            self._s[0][4] = 0 #Has picked up gold

            #For 4 by 4 grid 0 = Nothing, 1 = pit, 2 = wumpus, 3 = gold
            self._s[1][0][2] = 2 # Wumpus at x = 0, y = 2
            self._s[1][1][2] = 3 # Gold at x = 1, y = 2

            self._s[1][2][0] = 1 # Pit at x = 2, y = 0
            self._s[1][2][2] = 1 # Pit at x = 2, y = 2
            self._s[1][3][3] = 1 # Pit at x = 3, y = 3

        else :
            self._s = copy.deepcopy(s)

        if self.viewer: self.viewer.close()
        self.viewer = None

        if self.oberservable == "full":
            return np.array(np.append(np.reshape(self._s[1],[1,16]), np.reshape(self._s[0], [1, 5])))
        elif self.oberservable == "perceptions":
            return self.getPerceptions(False, False)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = 4 + 2
        world_height = 4 + 2
        xscale = screen_width/world_width
        yscale = screen_height/world_height
        agentwidth = 0.5
        agentheight = 0.5
        itemwidth = 0.5
        itemheight = 0.5

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xy = [(0.5,0.5),(0.5, 4 + 1.5), (4+ 1.5,4 + 1.5), (4 + 1.5,0.5)]
            xys = [ (x*xscale,y*yscale) for x,y in xy]
            self.surface = rendering.make_polyline(xys)
            self.surface.set_linewidth(4)
            self.viewer.add_geom(self.surface)

            xy = [(0.5,0.5),(4 + 1.5,0.5)]
            xys = [ (x*xscale,y*yscale) for x,y in xy]
            self.surface = rendering.make_polyline(xys)
            self.surface.set_linewidth(4)
            self.viewer.add_geom(self.surface)

            self.item_area = []
            self.item_trans = []
            self.item_pos = []
            k = 0
            for i in range(4):
                for j in range(4):
                    if self._s[1][i][j] == 2: # Check for wumpus
                        l,r,t,b = -itemwidth, itemwidth, itemheight, -itemheight
                        self.item_area.append(rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]))
                        self.item_area[k].set_color(0.0,0.8,0.0)
                        self.item_area[k].add_attr(rendering.Transform(translation=(0, 0)))
                        self.item_trans.append(rendering.Transform())
                        self.item_area[k].add_attr(self.item_trans[k])
                        self.viewer.add_geom(self.item_area[k])
                        self.item_pos.append([i, j])
                        k += 1

                    if self._s[1][i][j] == 1: # Check for pit
                        l,r,t,b = -itemwidth, itemwidth, itemheight, -itemheight
                        self.item_area.append(rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]))
                        self.item_area[k].set_color(0.8,0.8,0.8)
                        self.item_area[k].add_attr(rendering.Transform(translation=(0, 0)))
                        self.item_trans.append(rendering.Transform())
                        self.item_area[k].add_attr(self.item_trans[k])
                        self.viewer.add_geom(self.item_area[k])
                        self.item_pos.append([i, j])
                        k += 1

                    if self._s[1][i][j] == 3: # Check for gold
                        l,r,t,b = -itemwidth, itemwidth, itemheight, -itemheight
                        self.item_area.append(rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]))
                        self.item_area[k].set_color(0.8,0.8,0.0)
                        self.item_area[k].add_attr(rendering.Transform(translation=(0, 0)))
                        self.item_trans.append(rendering.Transform())
                        self.item_area[k].add_attr(self.item_trans[k])
                        self.viewer.add_geom(self.item_area[k])
                        self.item_pos.append([i, j])
                        k += 1

                l,r,t,b = -agentwidth, agentwidth, agentheight, -agentheight
                self.agent = rendering.FilledPolygon([(l,b), ((r+l)/2.0,t), (r,b)])
                self.agent.set_color(0.0,0.0,0.8)
                self.agent.add_attr(rendering.Transform(translation=(0, 0)))
                self.agent_trans = rendering.Transform()
                self.agent.add_attr(self.agent_trans)
                self.viewer.add_geom(self.agent)

        for k in range(len(self.item_trans)) :
            if self._s[1][int(self.item_pos[k][0])][int(self.item_pos[k][1])] == 0:
                self.item_trans[k].set_scale(0,0)
            else:
                self.item_trans[k].set_translation((self.item_pos[k][0] + 1) * xscale, (self.item_pos[k][1] + 1) * yscale)
                self.item_trans[k].set_scale(xscale,yscale)

        self.agent_trans.set_translation((self._s[0][0] + 1) * xscale, (self._s[0][1] + 1) * yscale)
        self.agent_trans.set_scale(xscale,yscale)
        self.agent_trans.set_rotation((-math.pi/2) * self._s[0][2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer: self.viewer.close()
