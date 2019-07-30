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
from profilehooks import profile
import logging

logger = logging.getLogger('aamas')
logger.setLevel(global_defs.debug_level)

#GLOBAL DEFINTIONS ABOUT THE AGENT_LIFTER.py
agent_state_def = namedtuple('alifter_state','tp name pos ALPHA')

class agent_lifter(Agent):
    def __init__(self,pos,tp,config_agent=None):
        super().__init__(pos,tp)
        self.tp = tp
        self.name = self.name+'_lifter_'+str(self.id)
        if not utils.check_within_boundaries(pos):
            warnings.warn("Init positions out of bounds",UserWarning)
            logger.warning("Init positions out of bounds")
        self.pos = global_defs.Point2D(pos[0],pos[1])

        if config_agent is not None:
            logger.debug("Initialized with config {}".format(config_agent))
            self.config_agent = config_agent
            self.ALPHA = config_agent.ALPHA
        else:
            self.ALPHA = global_defs.DEFAULT_ALPHA #default value of Alpha


        self.target = None
        self.load = False
        self.is_adhoc = False
        self.orientation = 2

        logger.info("agent_lifter.init initialized agent {}".format(self))

    def respond(self,observation: global_defs.obs) -> tuple:
        if not isinstance(observation,global_defs.obs):
            print(observation)
            raise Exception("Type checking exception")
        if self.tp<=0:
            raise Exception("Invalid Type Less than zero")
        else:
            targetIdx = observation.loadIndices[self.tp-1]

        logger.debug("agent_lifter.respond called for agent {} with observation {}".format(self,observation))
        target_pos = observation.allPos[targetIdx]


        #ALGO:
        #if neighbor-to-target:
            #Change action to load only.
            #No Softmax
        #else:
            #Do ASTAR over to the target.
            #if path_found:
                #Assign probability 1 to action accordingly.
                #Softmax
                #Zero to LOAD.
            #else:
                #Assign equal probabilities to all valid actions.
                #No Softmax
                #Zero to LOAD.
        action_mask = np.zeros(6,dtype=bool)
        action_mask[-2] = 1 #NoOp is always possible.
        action_probs = np.zeros(6)

        if utils.is_neighbor(self.pos,target_pos):
            action_probs[global_defs.Actions.LOAD]=1
            logger.debug("agent_lifter {} responded with LOAD action".format(self))

        else:
            #Every location except self and target.
            obstacles = []
            myidx = observation.myInd
            for idx,pos in enumerate(observation.allPos):
                if idx!=myidx and idx!=targetIdx:
                    obstacles.append(pos.as_tuple())

            #Get all valid movements based on current location.
            all_moves = global_defs.MOVES[:-1]
            for idx,move in enumerate(all_moves):
                new_pos = self.pos + move
                if utils.check_valid(new_pos,obstacles):
                    action_mask[idx]=1


            astar_solver = astar(self.pos.as_tuple(),target_pos.as_tuple(),obstacles, global_defs.GRID_SIZE,False)
            (path_found,path) = astar_solver.find_minimumpath()


            if path_found:
                #We are not next to the goal.
                movement = path[1]-path[0]
                planned_action = global_defs.MOVES_TO_ACTIONS[(movement[0],movement[1])]
                #planned_move = path[1].move
                action_probs[planned_action] =1

            action_probs = utils.softmax(action_probs,self.ALPHA)

            #Only retain valid actions
            action_probs*=action_mask

            #Renormalizing, since softmax has the ability to add values to non zero inputs.
            action_probs = action_probs/sum(action_probs)


        ##Post processing tips on actionprobs about SoftMax
        #Almost just like normalizing. The lower alpha is the higher the noise.
        # 20 - Normalizing , e-09 when input is zero
        # 10 - Normalizing , e-05 when input is zero.
        # 5 - Almost normalizing, e-03 when input is zero.
        # 4 - Noise - e-02 when input is zero.
        # 3 - Noise - 2xe-03 when input is zero
        # 1 - Noise - e-01
        # 0 - All are equal. Full Noise.

        # valid actions
        valid_actions = np.where(action_probs[:4] > 0)[0]
        for action in valid_actions:
            newPos = self.pos + global_defs.MOVES[action]
            if not utils.check_within_boundaries(newPos):
                raise Exception()
        action = np.random.choice(global_defs.Actions,p=action_probs)
        logger.debug("agent_lifter {} responded with aprobs {} and action {}".format(self,action_probs,action))
        return (action_probs,action)

    def act(self, proposal ,decision: bool) -> None:
        if decision==True:
            action_probs,action_idx = proposal
            logger.debug("agent_lifter {} acting on proposal {} with decision {}".format(self,proposal,decision))
            if action_idx==global_defs.Actions.NOOP or action_idx==global_defs.Actions.LOAD:
                #Do nothing.
                pass
            else:
                try:
                    move = global_defs.ACTIONS_TO_MOVES[action_idx]
                except:
                    print("DEBUG")
                self.pos += move
        else:
            logger.debug("agent_lifter {} not-acting on proposal {} with decision {}".format(self,proposal,decision))
        return

    def __copy__(self):
        state = self.__getstate__()
        new_agent_instance = agent_lifter(state.pos,state.tp)

        new_agent_instance.name += '_c'

        new_agent_instance.ALPHA = state.ALPHA

        return new_agent_instance

    def __setstate__(self, state):
        if not isinstance(state,agent_state_def):
            raise Exception("Recieved something else instead of State")
        self.pos = state.pos
        self.tp = state.tp
        self.ALPHA = self.ALPHA

    def __getstate__(self):
        state = agent_state_def(self.tp,self.name,self.pos,self.ALPHA)
        state = copy.deepcopy(state)
        return state

    def __repr__(self):
        pstr = ''
        pstr += 'Name: {} '.format(self.name)
        pstr += 'Tp: {} '.format(self.tp)
        pstr += 'Pos: {} '.format(self.pos)
        if self.target is not None:
            pstr += 'Target: {} '.format(self.target)

        return pstr


