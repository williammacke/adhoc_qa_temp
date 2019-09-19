import numpy as np
import matplotlib.pyplot as plt
from src.agents import agent
from src.astar import pyastar
from collections import namedtuple
from src import global_defs

#TODO:
"""
Leader: 

Modifications from original: 

Init: 

1) Generate a random order of stations and tools.  

At each step: 

1) If a question is asked, pause and answer it.  

2) Else: 

If neighboring a station: 

    If station task not completed: 

        If next agent is here: 

            Mark the task finished  

        else 

            Wait until next agent arrives. 

 Else 

        Navigate to the next station. 

3) Repeat until all stations are marked done.  

"""
#logger = logging.getLogger('aamas')
#logger.setLevel(global_defs.debug_level)

Agent_state = namedtuple('AgentState','type pos target')


class agent_leader(agent.AbstractAgent):
    def __init__(self,pos):
        super().__init__(pos,tp=None)
        self.pos = pos
        self.name = self.name+'_leader_'+str(self.id)
        if True:
            self.tp = agent.AgentType(global_defs.n_stations) #Initializing type from settings derived from global_defs.
        else:
            self.tp = tp

    def respond(self,observation: global_defs.obs)-> tuple:
        obs = observation
        target = self.tp.get_next_job_station()
        self.__target = target
        target_pos = obs.allPos[stationIndices[target]]
        obstacles = copy.deepcopy(obs.allPos).remove(self.pos)

        desired_action = None
        if utils.is_neighbor(self.pos,target_pos):
            #We are neighboring the station we want to work at.
            if utils.is_neighbor(target_pos,obs.allPos[adhocInd]):
                #Meaning, if the other agent is also neighoring the station, execute the work action
                desired_action = global_defs.Actions.WORK
                #now, if the other agent also executed a WORK action, and happens to have the same tool, then we can safely move to the next station. This change will happen if the environment approves the work action, since, for the action to run, the agent needs a way to know if the other agent also has thetool to operate in this station. This tool checking will be done by the environment.
            else:
                #Else, just wait.
                desired_action = global_defs.Actions.NOOP
        else:
            #Meaning we yet have to reach our target station.
            desired_action = None 

        proposal = utils.generate_proposal(self.pos,target_pos,obstacles,desired_action)
        return proposal
        
    def act(self,proposal,decision):
        _,action = proposal
        if decision is True:
            #Then we are allowed to execute the action.
            #First, apply the movement.
            self.pos += global_defs.ACTIONS_TO_MOVES[action]
            #logger.debug("Agent {} executed action: {}".format(self.name,action))
            if action == global_defs.Actions.WORK:
                #Which means we were approved to go ahead and do the work action, because the other agent had the right tool with it. It's time to move onto the next station.
                self.tp.set_status(self.__target) #mark the station's status as done. 
                #logger.debug("Agent {} finished target station id {}".format(self.name,self.__target))
        else:
            pass

    def set_tp(self,tp):
        self.tp = tp

    def __repr__(self):
        st = '{}_at_{}_withtp_{}_goingto_{}'.format(self.name,self.pos,self.tp,self.__target)
        return st

    def __copy__(self):
        """
        Make a copy of the agent. Copying the following is enough to make a good copy of the agent. We don't differentiate between deepcopy and copy. All are the same. 
        """
        new_tp = copy.deepcopy(self.tp)
        new_pos = copy.deepcopy(self.pos)
        new_agent = agent.AbstractAgent(new_pos,new_tp)
        return new_agent

    def __deepcopy__(self):
        return self.__copy__()

    def make_identical_to(self,new_agent):
        """
        This makes the agents' identical to each other. 
        """
        
        self.tp = copy.deepcopy(new_agent.tp)
        self.pos = copy.deepcopy(new_agent.pos)
        return True

        

