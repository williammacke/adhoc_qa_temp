import numpy as np
from enum import IntEnum
from recordclass import recordclass
from collections import namedtuple
#import pdb
import math
import logging



debug_level = logging.DEBUG
# create logger with 'spam_application'
logger = logging.getLogger('adhoc_qa')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('adhoc_qa.log')
fh.setLevel(logging.CRITICAL)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.CRITICAL)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

DEBUG = False

MAX_ITERS = 100

# These parameters are initialized when evironment is initialized.
GRID_SIZE = 0
N_STATIONS = 0

# Code already had tools in other files though it didn't exist. Should move to environment?
# TOOL_BASE = Point2D(0, 0)

#An observation should have all positions, my index to get my position from all of them, station indices, and finally, the most recent actions performed by all the agents in the arena.
obs = recordclass('obs','allActions allPos stationStatus stationInd')
LEADER_IDX = 0
ADHOC_IDX = 1
#obs_history = namedtuple('trajectory','listOfObservations')

BIAS_PROB = 1 # Leader is always optimal. This is the probability most optimal decision is made.


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self,other):
        x = self.x+other[0]
        y = self.y+other[1]
        return Point2D(x,y)

    def __sub__(self, other):
        "self-other"
        x = self.x - other[0]
        y = self.y - other[1]
        return Point2D(x,y)

    def __str__(self):
        return str((self.x,self.y))

    def __hash__(self):
        return (self.x,self.y).__hash__()

    def __eq__(self, other):
        if self.x==other[0] and self.y==other[1]:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __mul__(self, other):
        if isinstance(other,Point2D) or isinstance(other,tuple):
            return Point2D(self.x*other[0],self.y*other[1])
        else:
            return Point2D(self.x*other,self.y*other)

    def __rmul__(self, other):
        if isinstance(other,Point2D) or isinstance(other,tuple):
            return Point2D(self.x*other[0],self.y*other[1])
        else:
            return Point2D(self.x*other,self.y*other)

    def __copy__(self):
        return Point2D(self.x,self.y)

    def as_array(self):
        return np.array([self.x,self.y])

    def as_tuple(self):
        return (self.x,self.y)

    def manhattan_dist(self,other):
        diff = self - other
        return abs(diff[0])+abs(diff[1])
    def norm_dist(self,other):
        diff = self - other
        return math.sqrt((diff[0]**2)+(diff[1]**2))

Actions = IntEnum('Actions','RIGHT LEFT UP DOWN NOOP WORK',start=0)
Actions_list = list(Actions)

ACTIONS_TO_MOVES = {0:Point2D(1,0),1:Point2D(-1,0),2:Point2D(0,1),3:Point2D(0,-1),4:Point2D(0,0),5:Point2D(0,0)}
ACTIONS_SET = set(ACTIONS_TO_MOVES.keys())

MOVES_TO_ACTIONS = dict((move,action) for action,move in ACTIONS_TO_MOVES.items())
MOVES = [ele[1] for ele in ACTIONS_TO_MOVES.items()]
MOVES_SET = set(MOVES)

#Actions_8 = IntEnum('Actions_8','RIGHT LEFT UP DOWN RIGHT-UP RIGHT-DOWN LEFT-UP LEFT-DOWN NOOP')
#ACTIONS_TO_MOVES_8 = {0:Point2D(1,0),1:Point2D(-1,0),2:Point2D(0,1),3:Point2D(0,-1),4:Point2D(1,1),5:Point2D(1,-1),6:Point2D(-1,1),7:Point2D(-1,-1),8:Point2D(0,0)}
#MOVES_8 = ACTIONS_TO_MOVES_8.values()
