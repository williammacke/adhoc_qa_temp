import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from src import global_defs as gd
from src import utils
#import pdb
debug = gd.DEBUG
import random
from src.agents.agent import AgentType
import copy

from profilehooks import profile
config = namedtuple("cfg",'S k maxstep')
config_env = config(1000,1,150)

env_state_def = namedtuple('env_state_def','size n_stations sttn_pos all_actions is_terminal step_count config adhoc_state')

class environment():
    def __init__(self,size,sttn_positions,tools_pos,visualize=False,config_environment=config_env):

        """
        :param size: Dimension of the grid in which the environment is assumed to live.
        :param sttn_positions: Positions of the station inside the grid. The positions are in regular axes, and not numpy notations.
        """

        # Initialize global variables
        gd.GRID_SIZE = size
        gd.N_STATIONS = len(sttn_positions)

        self.size = size
        self.n_stations = len(sttn_positions)
        self.sttn_pos = sttn_positions
        self.tools_pos = tools_pos
        self.agents = [None, None] # Assumed only 2 agents, leader and adhoc respectively
        self.allActions = [None, None]

        self.communication_time_steps = []

        self.is_terminal = False
        self.step_count = 0
        self.visualize = visualize
        if self.visualize:
            import pygame
            import threading
            import src.levelbasedforaging_visualizer as vis_class
            self.vis_library = vis_class
            sttn_positions_copy = [pos.__copy__() for pos in self.sttn_pos]
            self.visualizer = vis_class.Visualizer(self.size,sttn_positions_copy,[],[])
            self.visualize_thread = threading.Thread(target=self.visualizer.wait_on_event)
            self.visualize_thread.start()

        if config_environment is not None:
            self.config = config_environment

        self.history = []

    def register_agent(self,agent):
        self.agents[gd.LEADER_IDX] = agent

    def register_adhoc_agent(self,adhoc_agent):
        """
        :param adhoc_agent: The adhoc agent's interface.
        :return:
        """
        self.agents[gd.ADHOC_IDX] = adhoc_agent

    def register_communication_time_steps(self, communication_time_steps):
        """
        :param communication_time_steps: a list of step_count ints that communication happens
        :return:

        Whenever communication happens, both agents do Action.NOOP.
        """
        self.communication_time_steps = communication_time_steps

    def generate_observation(self):
        agent_locs = [agent.pos for agent in self.agents]
        all_locs = agent_locs + self.tools_pos + self.sttn_pos

        leader_tp = self.agents[gd.LEADER_IDX].tp
        station_status_ordered = [AgentType.status.pending] * gd.N_STATIONS
        station_locs = self.sttn_pos
        for station, status in zip(leader_tp.station_order, leader_tp.station_work_status):
            station_status_ordered[station] = status

        station_ind = list(range(len(agent_locs) + len(self.tools_pos), len(all_locs)))
        obs = gd.obs(self.allActions, all_locs, station_status_ordered, station_ind, self.step_count)

        return obs

    # THIS IS MANISH'S CODE, BUT HE TREATED AND IMPLEMENTED THE OBSERVATIONS AS LEADER OBSERVATIONS
    # CHANGED: ADHOC AGENT NOW RECEIVES SAME OBSERVATIONS AS LEADER
    # def generate_adhoc_observation(self):
    #     """
    #     Return AdHoc observation. For Adhoc observation, you return the environment itself and not any specific observation.
    #     :return:
    #     """
    #     return self

    def update_vis(self):
        #TODO: Agent Orientation Attribute to Agents
        agent_positions = [agent.pos.__copy__() for agent in self.agents]
        agent_orientations = [agent.orientation for agent in self.agents]

        station_positions = [sttn_pos.__copy__() for sttn_pos in self.sttn_pos]
        self.update_event = self.vis_library.pygame.event.Event(self.visualizer.update_event_type,{
            'agents_positions':agent_positions,'agents_orientations':agent_orientations,'sttn_positions':station_positions
        })
        self.vis_library.pygame.event.post(self.update_event)

    def _step_dispatch(self):
        """
        Dispatches observations and collects agents' proposals for actions.
        :return:
        """
        agent_proposals = [] #Probability distributions over actions
        n_agents = len(self.agents)
        observation = self.generate_observation()
        for (agent_idx,agent) in enumerate(self.agents):

            # CHANGED: THIS ISN'T NEEDED SINCE ADHOC OBSERVATIONS HAS BEEN CHANGED TO BE SAME AS LEADER OBSERVATIONS
            # if agent_idx==n_agents-1:
            #     #Check if the last agent is adhoc or not.
            #     if agent.is_adhoc:
            #         #Adhoc agent
            #         curr_observation = self.generate_observation()
            #     else:
            #         #Not adhoc agent.
            #         curr_observation = self.generate_observation(agent_idx)

            proposal = agent.respond(observation)
            agent_proposals.append(proposal)

        if debug:
            try:
                assert(len(agent_proposals)==len(self.agents))
            except:
                print("Exception here")
                #pdb.set_trace()
            for agent_proposal,action_idx in agent_proposals:
                try:
                    np.testing.assert_almost_equal(sum(agent_proposal),1,5)
                    assert(isinstance(agent_proposal,np.ndarray))
                    assert(action_idx<6 and action_idx>=0)
                except:
                    #pdb.set_trace()
                    print("exception here")

        return agent_proposals,observation

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
                    #pdb.set_trace()
                    print("exception")
        decisions = []
        proposals = []

        n_agents = len(self.agents)
        #random_agent_order = random.sample(range(n_agents),n_agents) #Randomize all agent's priority.
        agent_order = range(n_agents)

        for agent_idx in agent_order:
            #Checks go in here, if the proposal can be valid or not.
            proposal = agent_proposals[agent_idx]
            decision = self._proposal_check(agent_idx,proposal)
            decisions.append(decision)
            proposals.append(proposal)
            self.agents[agent_idx].act(proposal,decision)

        if debug:
            for decision in decisions:
                assert(isinstance(decision,bool))
            assert(len(decisions)==len(self.agents))

        self.allActions = [prop[1] if dec else gd.Actions.NOOP for prop, dec in zip(proposals, decisions)]
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
                #pdb.set_trace()
                print("error")

        if action_idx == gd.Actions.NOOP:
            #Approve NOOP always.
            decision = True
            return True

        elif action_idx == gd.Actions.WORK:
            #Approve a load decision if it is near an station.

            for sttn_id, sttn_pos in enumerate(self.sttn_pos):
                if utils.is_neighbor(self.agents[agent_idx].pos,sttn_pos) and self.agents[gd.ADHOC_IDX].tool == sttn_id:
                    #If it neighbors any of the stations, then return True
                    decision = True
                    return decision

            #It means it isn't a neighbor to any station.
            decision = False
            return False

        else:
            #Look what can be approved.
            action_result = self.agents[agent_idx].pos+(gd.ACTIONS_TO_MOVES[action_idx])

            # THESE CHECKS NO LONGER NEEDED. AGENTS CAN OVERLAP STATIONS AND OTHER AGENTS.
            #Check it isn't moving into a invalid location. Part 1: Stations
            #for sttn_pos in self.sttn_pos:
            #    if sttn_pos == action_result:
            #        decision = False
            #        return decision

            #Check it isn't moving into a invalid location. Part 1: Agents
            #for agent in self.agents:
            #    if agent.pos == action_result:
            #        decision = False
            #        return decision

            #Allclear, send a yes.
            decision = True
            return decision

    def step(self):
        """
        Performs one step
        a1 = agent_lifter.agent_lifter(agent_pos[0], 2)
        a2 = agent_lifter.agent_lifter(agent_pos[1], 2)
        a3 = agent_adhoc.agent_adhoc(a2.pos)

        env = environment.environment(gd.GRID_SIZE, sttn_pos, False)

        env.register_agent(a1)
        env.register_agent(a2)

        n_steps = 0
        # print("Target Distance: {}".format(target_dist))
        # print("Station location {}".format(station_location))
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
        self.step_count+=1

        if self.step_count not in self.communication_time_steps:
            agent_proposals,observation = self._step_dispatch()
            decisions = self._step_decide_and_apply(agent_proposals)
            self.history.append((observation,agent_proposals,decisions))
        else:
            # Give all actions Action.NOOP for communication. Also adds to history just for consistency
            agent_proposals = [(np.zeros(len(gd.Actions),dtype='float'), gd.Actions.NOOP)] * 2
            observation = self.generate_observation()
            decisions = self._step_decide_and_apply(agent_proposals)
            self.history.append((observation, agent_proposals, decisions))

            # Update adhoc knowledge: give adhoc agent the current target of the leader agent
            leader_target = self.agents[gd.LEADER_IDX].tp.get_current_job_station()
            self.agents[gd.ADHOC_IDX].knowledge.update_knowledge_from_qa([leader_target])

        (self.is_terminal,reward) = self.check_for_termination()
        if self.visualize:
            self.update_vis()
        return (self.is_terminal,reward)

    def check_for_termination(self):
        """
        Checks for ending criterion and sends reward.

        Since the task is dependent on leader agent finishing the set of stations, we ought to wait until the fist agent signals completition. There is no other way to see if it is finished.

        """
        leader = self.agents[gd.LEADER_IDX]
        terminated = False
        reward = 1
        if AgentType.status.pending not in leader.tp.get_status():
            terminated = True
        return (terminated, 1)

    # TODO: THIS FUNCTION DOESN'T WORK. ADHOC_AGENT HAS NO COPY FUNCTION
    #       EVERYTHING ELSE OF COPY SHOULD BE FINE
    def __copy__(self):
        selfstate = self.__getstate__()
        new_env = self.__init_from_state__(selfstate)
        return new_env

    def __init_from_state__(self,selfstate):
        new_env = environment(selfstate.size,selfstate.sttn_pos,False,selfstate.config)
        # OLD CODE
        # for agent in self.agents:
        #     new_agent = agent.__copy__()
        #     new_env.register_agent(new_agent)

        # ASSUME ONLY LEADER AND ADHOC AGENTS
        new_env.register_agent(self.agents[gd.LEADER_IDX].__copy__())
        new_env.register_adhoc_agent(self.agents[gd.ADHOC_IDX].__copy__())

        new_env.__setstate__(selfstate)
        return new_env

    def __getstate__(self):
        # Only need to get state of adhoc agent. For leader agent, __copy__() already fully preserves state
        adhoc_state = self.agents[gd.ADHOC_IDX].__getstate__()
        curr_state = env_state_def(self.size,self.n_stations,self.sttn_pos,self.allActions,\
                self.is_terminal,self.step_count,self.config,adhoc_state)
        curr_state = copy.deepcopy(curr_state)
        return curr_state

    def __setstate__(self,s):
        s = copy.deepcopy(s)
        if not isinstance(s,env_state_def):
            raise Exception("Invalid state sent")
        self.n_stations = s.n_stations
        self.sttn_pos = s.sttn_pos
        self.allActions = s.all_actions
        self.is_terminal = s.is_terminal
        self.step_count = s.step_count
        self.config = s.config
        # Set state for only the adhoc agent. State of leader agent already set with __copy__()
        self.agents[gd.ADHOC_IDX].__setstate__(ast)
        return

    def __repr__(self):
        pstr = ''
        pstr += 'n_agents: {} '.format(len(self.agents))
        pstr += 'n_stations: {} '.format(self.n_stations)
        pstr += 'curr_step: {} '.format(self.step_count)
        pstr += 'Agents: {} '.format([id(agent) for agent in self.agents])
        return pstr
