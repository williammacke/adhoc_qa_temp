from src.agents.agent import Agent
from .. import global_defs
import numpy as np
from src.astar.pyastar import astar
from collections import namedtuple
import copy
import pdb
import ipdb
from src import utils
import warnings
import logging

config_agent = namedtuple('config_agent','ALPHA')
#GLOBAL DEFINTIONS ABOUT THE AGENT_LIFTER.py
agent_state_def_random = namedtuple('arandom_state','pos name')
logger = logging.getLogger('aamas')


class agent_random(Agent):
    def __init__(self,pos):

        super().__init__(pos,-1)
        self.tp = -1
        self.name = self.name+'_random_'+str(self.id)
        if ((pos[0]<0 or pos[0]>global_defs.GRID_SIZE-1) and (pos[1]<0 or pos[1]>global_defs.GRID_SIZE-1)):
            warnings.warn("Init positions out of bounds",UserWarning)
        self.pos = global_defs.Point2D(pos[0], pos[1])
        logger.debug("agent_random created new agent {}".format(self))
        self.is_adhoc = False

    def respond(self,observation):
        #Just respond with a random action from whatever is possible.
        #This agent is supposed to find all valid actions and output a random prob distribution over actions
        #as valid ones.

        #Note: Load is also a possible action.

        #Premature optimization is the rootcause of all evil.
        #OPtimization ideas from Guido URL.

        logger.debug("agent_random {} asked to respond with {}".format(self,observation))
        action_probs = np.zeros(len(global_defs.Actions))

        objectIdxs = observation.loadIndices
        #Todo optimize .remove operation
        agentIdxs = list(range(len(observation.allPos)))
        agentIdxs.remove(observation.myInd)
        #ipdb.set_trace()
        for idx in objectIdxs:
            agentIdxs.remove(idx)

        for aidx,move in enumerate(global_defs.MOVES[:-1]): #Only movement actions are being tested now.
            newPos = self.pos + move
            reject_action = False

            #Check if within bounds
            if not utils.check_within_boundaries(newPos):
                continue

            #Check if hitting any object or other agent.
            for idx in objectIdxs:
                if newPos==observation.allPos[idx]:
                    reject_action = True
                    action_probs[global_defs.Actions.LOAD]=1
                    #if we are hitting an object, then even load action is possible.
                    break

            if reject_action:
                #Action probability for this was zero anyway.
                continue

            #Check if hitting other agent.
            for idx in agentIdxs:
                if newPos == observation.allPos[idx]:
                    reject_action=True
                    break

            if reject_action:
                #Action probability for this was zero anyway.
                continue


            #If we are still inside a loop, then it is a one.
            action_probs[aidx] = 1

        action_probs[-2] =1 #NoOp is always a valid action.
        action_probs/=sum(action_probs)

        #valid actions
        valid_actions = np.where(action_probs>0)[0]
        for action in valid_actions[:4]:
            newPos = self.pos + global_defs.MOVES[action]
            if not utils.check_within_boundaries(newPos):
                logger.critical("{} somehow chose invalid action {}".format(self,action))
                raise Exception()

        action_idx = np.random.choice(global_defs.Actions,p=action_probs)
        logger.debug("{} responded with {} {}".format(self,action_probs,action_idx))
        return (action_probs, action_idx)


    def act(self,proposal, decision: bool):
        #1) If the decision is true, just forward the movement.
        #2) Otherwise just keep the action space and don't do anything.
        logger.debug("{} asked to act on proposal {} with decision {}".format(self,proposal,decision))
        if decision == True:
            action_probs, action_idx = proposal
            if action_idx == global_defs.Actions.NOOP or action_idx == global_defs.Actions.LOAD:
                # Do nothing.
                pass
            else:
                move = global_defs.ACTIONS_TO_MOVES[action_idx]
                self.pos += move

        return

    def __copy__(self):
        na = agent_random(copy.deepcopy(self.pos))
        na.name += '_c'
        return na

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def __getstate__(self):
        s = agent_state_def_random(self.pos,self.name)
        return copy.deepcopy(s)

    def __setstate__(self, state):
        s = copy.deepcopy(state)
        self.pos = s.pos
        return

    def __repr__(self):
        retstr = 'Name: {} '.format(self.name)
        retstr += 'Tp: {} '.format(self.tp)
        retstr += 'Pos: {} '.format(self.pos)
        return retstr

