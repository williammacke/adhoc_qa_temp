from src.agents.agent import AbstractAgent
from src import global_defs as gd
import numpy as np
from collections import namedtuple
from enum import Enum
import copy
from src import utils
import warnings
import logging
from src.agents.adhoc_utils import Knowledge, inference_engine

logger = logging.getLogger('aamas')
logger.setLevel(gd.debug_level)

adhoc_agent_state = namedtuple('AgentState','type pos target')

class agent_adhoc(AbstractAgent):
    def __init__(self,pos):
        super().__init__(pos,tp=-2)
        self.pos = pos
        self.name = self.name+'_adhoc'+str(self.id)

        # logger.debug("Adhoc agent initialized  {}".format(self))
        if ((pos[0]<0 or pos[0]>gd.GRID_SIZE-1) or (pos[1]<0 or pos[1]>gd.GRID_SIZE-1)):
            # warnings.warn("Init positions out of bounds",UserWarning)
            logging.warning("Init positions out of bounds")
        self.is_adhoc=True
        self.tool = None #Start off with no tool.
        self.knowledge = Knowledge()
        self.p_obs = None
        self.p_obs_temp = None # Holds previous observation temporarily in case action not approved by environment
        # warnings.WarningMessage("Make sure tracking agent is registered")


    def register_tracking_agent(self,tagent):
        self.tracking_agent = tagent

    def get_remaining_stations(self,cobs):
        stations_left = []
        for sttnidx in range(gd.N_STATIONS):
            if cobs.stationStatus[sttnidx] is False:
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
        if obs.timestep == 0:
            #If it's the first timestep, we have no clue.
            self.tracking_stations = self.get_remaining_stations(obs)
            self.inference_engine = inference_engine(self.tracking_agent,self.tracking_stations)

            target_station = np.random.choice(self.tracking_stations) #pick a station at random.

        else:
            curr_k_id = self.knowledge.get_current_job_station_id()
            #Checking what knowledge we have.
            if (self.knowledge.source[curr_k_id]==Knowledge.origin.Answer):

                #Then we simply work on the station because we have an answer telling us that that's the station to work on.
                target_station = self.knowledge.station_order[curr_k_id]

            elif (self.knowledge.source[curr_k_id] == None):
                #which means we just finished a station in the last time-step. This calls for re-initalizing the inference_engine
                self.tracking_stations = self.get_remaining_stations(obs)
                self.inference_engine = inference_engine(self.tracking_agent,self.tracking_stations)
                target_station = np.random.choice(self.tracking_stations)

            elif (self.knowledge.source[curr_k_id]==Knowledge.origin.Inference):
                #Which means we have been working on a inference for a station.
                target_station = self.inference_engine.inference_step(self.p_obs,obs)
                self.knowledge.update_knowledge_from_inference(target_station)
                # warnings.WarningMessage("Provision resetting inference_engine when a station is finished")

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
                destination = gd.TOOL_BASE
        else:
            destination = gd.TOOL_BASE

        if utils.is_neighbor(self.pos,destination):
            if destination == gd.TOOL_BASE:
                #We are at the base to pick up a tool.
                desired_action = gd.Actions.NOOP
                self.tool = target_station
            else:
                #we are the station to work.
                desired_action = gd.Actions.WORK
        else:
            #Navigate to destination.
            desired_action = None

        obstacles = copy.deepcopy(obs.allPos)
        obstacles.remove(self.pos)
        proposal = utils.generate_proposal(self.pos,destination,obstacles,desired_action)
        return proposal

    def act(self, proposal, decision):
        if decision is True:
            #If the decision was to work, then we have some bookkeeping to do.
            _,action = proposal
            if action == gd.Actions.WORK:
                #We have been approved to work, station work is finished.
                #Signal Knowledge that the work is finished.
                curr_k_id = self.knowledge.get_current_job_station_id()
                # self.knowledge.
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
