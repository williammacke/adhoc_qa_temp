from src.agents.agent import AbstractAgent
from src.agents import agent
from src import environment as env
import numpy as np
from collections import namedtuple
from enum import Enum
import copy
from src import utils
import warnings
from src.agents.adhoc_utils import Knowledge, inference_engine


adhoc_agent_state = namedtuple('AgentState','type pos target')

class agent_adhoc(AbstractAgent):
    def __init__(self,pos):
        super().__init__(pos,tp=-2)
        self.pos = pos
        self.name = self.name+'_adhoc'+str(self.id)

        if ((pos[0]<0 or pos[0]>env.GRID_SIZE-1) or (pos[1]<0 or pos[1]>env.GRID_SIZE-1)):
            # warnings.warn("Init positions out of bounds",UserWarning)
            logging.warning("Init positions out of bounds")
        self.is_adhoc=True
        self.tool = None #Start off with no tool.
        self.knowledge = Knowledge()
        self.certainty = False # Set True if only one station has max likelihood. Lets environment know not to allow query.
        self.p_obs = None
        self.p_obs_temp = None # Holds previous observation temporarily in case action not approved by environment
        # warnings.WarningMessage("Make sure tracking agent is registered")


    def register_tracking_agent(self,tagent):
        self.tracking_agent = tagent
        self.inference_engine = inference_engine(self.tracking_agent,list(range(env.N_STATIONS)))

    def get_remaining_stations(self,cobs):
        stations_left = []
        for sttnidx in range(env.N_STATIONS):
            if cobs.stationStatus[sttnidx]  == agent.AgentType.status.pending:
                #This means this station hasn't been closed yet.
                stations_left.append(sttnidx)
        return np.array(stations_left)

    def respond(self,obs):
        """
        #First retrieve which station is next.
          -Go to knowledge and ask which one we are working on right now and what's the source. If we have it from QA, then skip inference.
          -If we have it from QA, perform inference, and get the latest estimate.
        #Then navigate.
          - If we have the tool, simply go forward.
          - Else, go to get the tool first.
        """
        self.p_obs_temp = obs

        curr_k_idx = self.knowledge.get_current_job_station_idx()
        #Checking what knowledge we have.
        if (self.knowledge.source[curr_k_idx]==Knowledge.origin.Answer):
            #Then we simply work on the station because we have an answer telling us that that's the station to work on.
            target_station = self.knowledge.station_order[curr_k_idx]

        elif (self.knowledge.source[curr_k_idx] == None):
            #which means we just started or finished a station in the last time-step. This calls for re-initalizing the inference_engine
            self.tracking_stations = self.get_remaining_stations(obs)
            self.inference_engine = inference_engine(self.tracking_agent,self.tracking_stations)
            target_station = np.random.choice(self.tracking_stations)
            self.knowledge.update_knowledge_from_inference(target_station)

        elif (self.knowledge.source[curr_k_idx]==Knowledge.origin.Inference):
            #Which means we have been working on a inference for a station.
            target_station, certainty = self.inference_engine.inference_step(self.p_obs,obs)
            self.knowledge.update_knowledge_from_inference(target_station)
            if certainty:
                self.certainty = certainty

        else:
            #it should never come to this.
            raise Exception("Some mistake around")

        """
        Okay, now that we know which station we should be headed to, we need to ensure the nitty-gritty details.
        Do we have a tool?
             If yes,
                if it matches our target station:
                     destination: station
                else:
                     destination: base
        else:
             destination: base

        Are we near our destination?
             Yes:
                Is it the base?
                    Pick up the tool.
                else:
                    execute work action.
             No:
                keep moving.
        """
        if self.tool is not None:
            if self.tool == target_station:
                destination = obs.allPos[obs.stationInd[target_station]]
            else:
                destination = obs.allPos[env.TOOLS_IDX]
        else:
            destination = obs.allPos[env.TOOLS_IDX]

        if utils.is_neighbor(self.pos,destination):
            if destination == obs.allPos[env.TOOLS_IDX]:
                #We are at the base to pick up a tool.
                desired_action = env.Actions.NOOP
                self.tool = target_station
            else:
                #we are the station to work.
                if utils.is_neighbor(destination,obs.allPos[env.LEADER_IDX]):
                    #Meaning, if the other agent is also neighoring the station, execute the work action
                    desired_action = env.Actions.WORK
                else:
                    #Else, wait for other agent to get to station
                    desired_action = env.Actions.NOOP
        else:
            #Navigate to destination.
            desired_action = None

        # obstacles = copy.deepcopy(obs.allPos)
        # obstacles.remove(self.pos)
        obstacles = []
        proposal = utils.generate_proposal(self.pos,destination,obstacles,desired_action)
        return proposal

    def act(self, proposal, decision):
        if decision is True:
            #If the decision was to work, then we have some bookkeeping to do.
            _,action = proposal
            self.pos += env.ACTIONS_TO_MOVES[action]
            if action == env.Actions.WORK:
                #We have been approved to work, station work is finished.
                #Signal Knowledge that the work is finished.
                curr_k_idx = self.knowledge.get_current_job_station_idx()
                self.knowledge.set_status(curr_k_idx)
            self.p_obs = self.p_obs_temp

    def __repr__(self):
        st = '{}_at_{}_goingto_{}'.format(self.name,self.pos,self.knowledge.get_current_job_station())
        return st

    # def __copy__(self):


    # def __getstate__(self):
    #     pass
    #
    # def __setstate__(self):
    #     pass
