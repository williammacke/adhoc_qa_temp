from src.agents.agent import Agent
from src import global_defs
import numpy as np
from src.agents import agent_random
from collections import namedtuple
import copy
import pdb
import ipdb
from src import utils
import warnings
from src.agents import agent_lifter
import logging

logger = logging.getLogger('aamas')
logger.setLevel(global_defs.debug_level)

#GLOBAL DEFINTIONS ABOUT THE AGENT_LIFTER.py
agent_adhoc_state_def = namedtuple('adhoc_state','tp name pos')

class agent_adhoc(Agent):
    def __init__(self,pos,tp=-2):
        super().__init__(pos,tp)
        self.tp = tp
        self.pos = pos
        self.name = self.name+'_adhoc'+str(self.id)

        logger.debug("Adhoc agent initialized  {}".format(self))
        if ((pos[0]<0 or pos[0]>global_defs.GRID_SIZE-1) and (pos[1]<0 or pos[1]>global_defs.GRID_SIZE-1)):
            warnings.warn("Init positions out of bounds",UserWarning)
            logging.warning("Init positions out of bounds")

        self.posterior_probability = np.zeros((global_defs.MAX_ITERS+1,global_defs.N_TYPES))
        self.posterior_probability[0]=np.ones(global_defs.N_TYPES,dtype='float')/(global_defs.N_TYPES)
        self.prior_probability = np.ones(global_defs.N_TYPES)/(global_defs.N_TYPES)
        self.likelihood = np.zeros((global_defs.MAX_ITERS+1,global_defs.N_TYPES))
        self.post_idx = 0

        logger.debug("Initializing exemplar agents")
        self.agent_exemplars = [agent_lifter.agent_lifter(global_defs.Point2D(0,0),i) for i in range(1,global_defs.N_TYPES+1)]
        self.is_adhoc=True
        self.orientation = 0
        self.prev_action = global_defs.Actions.NOOP

    def respond(self,observation):
        'Here observation is environment'
        obs_env = observation
        logger.debug("agent_adhoc.respond called with observation {}".format(observation))
        self.curr_step = obs_env.step_count

        #If we previous execute LOAD action, then execute the same now.

        if True:
            if self.curr_step == 0:
                post = self.prior_probability
                obs = obs_env.generate_observation(1)
            else:
                latest_info = obs_env.history[-1]
                (dispatched_observations,proposals,decisions) = latest_info

                #Get action
                if decisions[0] == True:
                    #Then we know for sure the action that was executed.
                    action = proposals[0][1]

                else:
                    action = global_defs.Actions.NOOP

                #Get the likelihood of this action for each type.
                obs = dispatched_observations[0] #Right now only one observation anyway.
                for agent in self.agent_exemplars:
                    #First set the example agent's state similar to the true agent's state.
                    agent.pos = obs.allPos[obs.myInd]
                    p,a = agent.respond(obs)
                    #print(agent.pos,agent.tp,agent.target)
                    #print(p,a,p[action])
                    self.likelihood[self.curr_step][agent.tp-1] = p[action]
                logger.info("agent_adhoc.respond likelihood computed to be {}".format(self.likelihood[self.curr_step]))

                post = self.posterior_probability[self.curr_step-1]*self.likelihood[self.curr_step]

            post = post/np.sum(post)
            logger.info("agent_adhoc.respond posterior update to {}".format(post))

            self.posterior_probability[self.curr_step]=post
            self.post_idx += 1

            self.tp_estimate= np.argmax(self.posterior_probability[self.curr_step])+1
            #logger.debug("agent_adhoc.respond calling MCTS_respond with observation {} and type estimate".format(obs_env,tp_estimate))
            obs = obs_env.generate_observation(1)
            target = obs.allPos[obs.loadIndices[self.tp_estimate-1]]

            #Check all directions and sort them within the order of closest to farthest. Execute LOAD if near an object.
            if utils.is_neighbor(self.pos,target):
                chosen_action = global_defs.Actions.LOAD
            else:
                all_valid_actions = []
                dists = []
                all_obstacles = copy.deepcopy(obs.allPos)
                all_obstacles.remove(self.pos)
                for action in global_defs.Actions_list[:-2]:
                    #Check if the movement is valid.
                    newPos = self.pos + global_defs.ACTIONS_TO_MOVES[action]
                    is_valid = utils.check_valid(newPos,all_obstacles)
                    if is_valid:
                        all_valid_actions.append([action,newPos])
                        dists.append(newPos.norm_dist(target))

                #Do we have legal actions to chose from?
                if len(all_valid_actions) == 0:
                    chosen_action = global_defs.Actions.NOOP
                else:
                    #Choose action that results in the closest of the both.
                    min_dist_actions_idxs = np.where(dists==np.min(dists))[0]
                    random_min_dist_action = all_valid_actions[np.random.choice(min_dist_actions_idxs)][0]
                    chosen_action = random_min_dist_action
                    #Adding noise.
                    if np.random.random()>0.1:
                        pass
                    else:
                        chosen_action = global_defs.Actions.NOOP

        #logger.critical("agent_adhoc.respond type estimated to {} ap {}".format(self.tp_estimate,obs_env.agents[0].tp))
        #logger.critical("A1 {}".format(obs_env.agents[0]))
        #logger.critical("A2 {}".format(obs_env.agents[1]))

        if self.prev_action is not None and self.prev_action==global_defs.Actions.LOAD:
            chosen_action = global_defs.Actions.LOAD

        proposal = np.zeros(6)
        proposal[chosen_action]=1
        return proposal,chosen_action

    def reset_belief(self,n_steps_backward):
        #if we last executed load action, then it is pointless anyway.
        if self.prev_action==global_defs.Actions.LOAD:
            pass
        else:
            self.posterior_probability[self.curr_step-n_steps_backward].fill(1/len(self.agent_exemplars))
            for i in range(n_steps_backward):
                prior = self.posterior_probability[self.curr_step-n_steps_backward+i]
                likelihood = self.likelihood[self.curr_step-n_steps_backward+i+1]
                posterior = prior*likelihood
                posterior /= np.sum(posterior)
                self.posterior_probability[self.curr_step-n_steps_backward+i+1] = posterior
        #otherwise, we need to push to the

    def act(self,proposal, decision: int):
        #Should we make this analogous to respond?
        #We got the previous agent's proposal and decision.
        if decision == True:
            #then we can actually see the action.
            action = proposal[1]
        else:
            action = global_defs.Actions.NOOP

        self.prev_action = copy.copy(action)

        #my_action = self.mcts_obj.choose_action(action)
        if action!=global_defs.Actions.LOAD:
            self.pos += global_defs.ACTIONS_TO_MOVES[action]
        return

    def __copy__(self):
        #Copying an adhoc agent results in creating a random agent.
        logger.debug("agent_adhoc.copy copying agent with state {}".format(self.pos))
        newagent = agent_random.agent_random(self.pos)
        newagent.name+='_c'

    def __getstate__(self):
        state = agent_adhoc_state_def(-1,self.name,self.pos)
        logger.debug("agent_adhoc.getstate retrieving state {}".format(state))
        return copy.deepcopy(state)

    def __setstate__(self, state):
        self.tp = state.tp
        self.pos = state.pos
        logger.debug("agent_adhoc.setstate setting state with {}".format(state))
        return

    def __repr__(self):
        pstr = ''
        pstr += 'Name: {} '.format(self.name)
        pstr += 'Tp: {} '.format(self.tp)
        return pstr

