import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from src import global_defs
from src import utils
import pdb
debug = global_defs.DEBUG
import random
from src.agents import agent_lifter
import copy

from profilehooks import profile
config = namedtuple("cfg",'S k maxstep')
config_env = config(1000,1,150)

env_state_def = namedtuple('env_state_def','size n_objects obj_pos is_terminal step_count config agent_state_list')

class environment():
    def __init__(self,size,obj_positions,visualize=False,config_environment=config_env):

        """
        :param size: Dimension of the grid in which the environment is assumed to live.
        :param obj_positions: Positions of the object inside the grid. The positions are in regular axes, and not numpy notations.
        """
        self.size = size
        self.n_objects = len(obj_positions)
        self.obj_pos = obj_positions
        self.agents = [] #initializing with a empty list of agents.

        self.is_terminal = False
        self.step_count = 0
        self.visualize = visualize
        if self.visualize:
            import pygame
            import threading
            import src.levelbasedforaging_visualizer as vis_class
            self.vis_library = vis_class
            obj_positions_copy = [pos.__copy__() for pos in self.obj_pos]
            self.visualizer = vis_class.Visualizer(self.size,obj_positions_copy,[],[])
            self.visualize_thread = threading.Thread(target=self.visualizer.wait_on_event)
            self.visualize_thread.start()

        if config_environment is not None:
            self.config = config_environment

        self.history = []

    def register_agent(self,agent):
        self.agents.append(agent)

    def register_adhoc_agent(self,adhoc_agent):
        """
        The adhoc agent should always be the last one.
        :param adhoc_agent: The adhoc agent's interface.
        :return:
        """
        self.agents.append(adhoc_agent)


    def generate_observation(self,agent_index):
        agent_locs = [agent.pos for agent in self.agents]
        object_locs = [pos for pos in self.obj_pos]
        all_locs = agent_locs+object_locs

        load_indices = range(len(self.agents),len(all_locs))
        obs = global_defs.obs(all_locs,agent_index,load_indices)

        return obs

    def generate_adhoc_observation(self):
        """
        Return AdHoc observation. For Adhoc observation, you return the environment itself and not any specific observation.
        :return:
        """
        return self

    def update_vis(self):
        #TODO: Agent Orientation Attribute to Agents
        agent_positions = [agent.pos.__copy__() for agent in self.agents]
        agent_orientations = [agent.orientation for agent in self.agents]

        object_positions = [obj_pos.__copy__() for obj_pos in self.obj_pos]
        self.update_event = self.vis_library.pygame.event.Event(self.visualizer.update_event_type,{
            'agents_positions':agent_positions,'agents_orientations':agent_orientations,'obj_positions':object_positions
        })
        self.vis_library.pygame.event.post(self.update_event)

    def _step_dispatch(self):
        """
        Dispatches observations and collects agents' proposals for actions.
        :return:
        """
        agent_proposals = [] #Probability distributions over actions
        n_agents = len(self.agents)
        observations=[]
        for (agent_idx,agent) in enumerate(self.agents):
            curr_observation = self.generate_observation(agent_idx)

            if agent_idx==n_agents-1:
                #Check if the last agent is adhoc or not.
                if agent.is_adhoc:
                    #Adhoc agent
                    curr_observation = self.generate_adhoc_observation()
                else:
                    #Not adhoc agent.
                    curr_observation = self.generate_observation(agent_idx)

            proposal = agent.respond(curr_observation)
            agent_proposals.append(proposal)
            observations.append(curr_observation)

        if debug:
            try:
                assert(len(agent_proposals)==len(self.agents))
            except:
                print("Exception here")
                pdb.set_trace()
            for agent_proposal,action_idx in agent_proposals:
                try:
                    np.testing.assert_almost_equal(sum(agent_proposal),1,5)
                    assert(isinstance(agent_proposal,np.ndarray))
                    assert(action_idx<6 and action_idx>=0)
                except:
                    pdb.set_trace()
                    print("exception here")

        return agent_proposals,observations

    def _step_decide_and_apply(self,agent_proposals):
        """
        Analyzes agent's proposals and decides approvals/denials.
        Then signals agent to apply the decision
        :type agent_proposals: (np.ndarray(4),int)
        :return decisions: list of True/False
        """

        if debug:
            assert(isinstance(agent_proposals,list))
            assert(len(agent_proposals)==len(self.agents))
            for proposal in agent_proposals:
                try:
                    assert(isinstance(proposal,tuple))
                    assert(isinstance(proposal[0],np.ndarray))
                except:
                    pdb.set_trace()
        decisions = []

        n_agents = len(self.agents)
        #random_agent_order = random.sample(range(n_agents),n_agents) #Randomize all agent's priority.
        random_agent_order = range(n_agents)

        for agent_idx in random_agent_order:
            #Checks go in here, if the proposal can be valid or not.
            proposal = agent_proposals[agent_idx]
            decision = self._proposal_check(agent_idx,proposal)
            decisions.append(decision)
            self.agents[agent_idx].act(proposal,decision) #Previous agent's proposal and decision are sent.


        if debug:
            for decision in decisions:
                assert(isinstance(decision,bool))
            assert(len(decisions)==len(self.agents))

        return decisions

    def _proposal_check(self,agent_idx:int ,proposal:tuple) -> bool:
        """
        Intended to check if this particular agent's proposal is acceptable or not.
        :param agent_idx: The agent_idx that suggested this proposal for itself
        :param proposal: The proposal  - (action_probs,action_idx)
        :return Yes/No: bool a response indicating yes/no.
        """

        action_probs,action_idx = proposal
        if debug:
            if action_idx>5 or action_idx<0:
                pdb.set_trace()

        if action_idx == global_defs.Actions.NOOP:
            #Approve NOOP always.
            decision = True
            return True

        elif action_idx == global_defs.Actions.LOAD:
            #Approve a load decision if it is near an object.

            for obj_pos in self.obj_pos:
                if utils.is_neighbor(self.agents[agent_idx].pos,obj_pos):
                    #If it neighbors any of the objects, then return True
                    decision = True
                    return decision

            #It means it isn't a neighbor to any object.
            decision = False
            return False

        else:
            #Look what can be approved.
            action_result = self.agents[agent_idx].pos+(global_defs.ACTIONS_TO_MOVES[action_idx])

            #Check it isn't moving into a invalid location. Part 1: Objects
            for obj_pos in self.obj_pos:
                if obj_pos == action_result:
                    decision = False
                    return decision

            #Check it isn't moving into a invalid location. Part 1: Agents
            for agent in self.agents:
                if agent.pos == action_result:
                    decision = False
                    return decision

            #Allclear, send a yes.
            decision = True
            return decision

    def step(self):
        """
        Performs one ste       a1 = agent_lifter.agent_lifter(agent_pos[0], 2)
        a2 = agent_lifter.agent_lifter(agent_pos[1], 2)
        a3 = agent_adhoc.agent_adhoc(a2.pos)

        env = environment.environment(global_defs.GRID_SIZE, obj_pos, False)

        env.register_agent(a1)
        env.register_agent(a2)

        n_steps = 0
        # print("Target Distance: {}".format(target_dist))
        # print("Object location {}".format(object_location))
        print("A1 {}".format(a1.pos))
        print("A2 {}".format(a2.pos))
        while (not env.is_terminal):
            is_terminal, reward = env.step()
            # ipdb.set_trace()
            time.sleep(0.1)
            tp_estimate = a3.respond(env)
            print("TPESTIMATE STEP", tp_estimate, env.step_count)
            if env.step_count == 5:
                a1.tp = 1
            n_steps += 1
            # print(n_steps)
            # print("A1 {}".format(a1.pos))
            # print("A2 {}".format(a2.pos))
            # if is_terminal:
            # print("Reward {}".format(reward))
        if is_terminal:
            print("Terminal Reward {}".format(reward))
            print("N_iters {}".format(n_steps))p of the simulation.
        :return:
        """
        agent_proposals,observations = self._step_dispatch()
        decisions = self._step_decide_and_apply(agent_proposals)
        self.step_count+=1
        (self.is_terminal,reward) = self.check_for_termination(agent_proposals,decisions)
        self.history.append((observations,agent_proposals,decisions))
        if self.visualize:
            self.update_vis()

        return (self.is_terminal,reward)

    def check_for_termination(self,agent_proposals,decisions):
        """
        Checks for ending criterion and sends reward.
        Reward structure = S-kT if successful, -kT if failed.
        return: tuple(is_terminal, reward). The reward is None if the state isn't terminal.

        """

        object_neighborhoods=[]
        object_neighbor_count=[]

        S = self.config.S
        k = self.config.k
        T = self.step_count
        maxsteps = global_defs.MAX_ITERS
        for obj_pos in self.obj_pos:
            curr_obj_neighborhood = []
            for agent,proposal,decision in zip(self.agents,agent_proposals,decisions):
                #An object counts as being in the neighborhood if it has the load action.
                proposed_action = proposal[1]
                if (utils.is_neighbor(obj_pos,agent.pos) and (proposed_action == global_defs.Actions.LOAD) and (decision == True)):
                    curr_obj_neighborhood.append(True)
                else:
                    curr_obj_neighborhood.append(False)
            object_neighborhoods.append(curr_obj_neighborhood)
            object_neighbor_count.append(sum(curr_obj_neighborhood))
        #print(object_neighborhoods)

        #Specific for our case, reward generation.
        if (self.n_objects>=2) and (len(self.agents)==2):
            #Either both agents are at one location.
            if 2 in object_neighbor_count:
                self.is_terminal = True
                reward = (S - (k*T))
                return (True,reward)

            #Or both are not at one location.
            elif object_neighbor_count.count(1)== 2:
                self.is_terminal = True
                reward = (0-(k*T))
                return (True,reward)

            #Both agents are not near the objects. So don't end the simulation yet if we still have steps left to go.
            else:
                if self.step_count == maxsteps:
                    self.is_terminal = True
                    reward = (0-(k*T))
                    return (True,reward)
                else:
                    self.is_terminal = False
                    return (False,None)

        elif len(self.agents)==2 and (self.n_objects==1):
            #If both agents are surrounding the object.
            if sum(object_neighborhoods[0])==2:
                self.is_terminal = True
                reward = (S - (k*T))
                return (True,reward)
            else:
                self.is_terminal = False
                reward = (0 - (k*T))
                return (False,reward)

    def __copy__(self):
        selfstate = self.__getstate__()
        new_env = self.__init_from_state__(selfstate)
        return new_env

    def __init_from_state__(self,selfstate):
        new_env = environment(selfstate.size,selfstate.obj_pos,False,selfstate.config)
        for agent in self.agents:
            new_agent = agent.__copy__()
            new_env.register_agent(new_agent)

        new_env.__setstate__(selfstate)
        return new_env

    def __getstate__(self):
        asl = [agent.__getstate__() for agent in self.agents]
        curr_state = env_state_def(self.size,self.n_objects,self.obj_pos,self.is_terminal,self.step_count,self.config,asl)
        curr_state = copy.deepcopy(curr_state)
        return curr_state

    def __setstate__(self,s):
        s = copy.deepcopy(s)
        if not isinstance(s,env_state_def):
            raise Exception("Invalid state sent")
        self.n_objects = s.n_objects
        self.obj_pos = s.obj_pos
        self.is_terminal = s.is_terminal
        self.step_count = s.step_count
        self.config = s.config
        for agent,ast in zip(self.agents,s.agent_state_list):
            agent.__setstate__(ast)
        return

    def __repr__(self):
        pstr = ''
        pstr += 'n_agents: {} '.format(len(self.agents))
        pstr += 'n_objects: {} '.format(self.n_objects)
        pstr += 'curr_step: {} '.format(self.step_count)
        pstr += 'Agents: {} '.format([id(agent) for agent in self.agents])
        return pstr



