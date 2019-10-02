"""
Defines abstract base clsas of what an Agent should be like. It can get more and more complicated as time passes, with complications inside agent models.
"""
from abc import ABC, abstractmethod, abstractproperty
from .. import global_defs as gd
import numpy as np
from itertools import count
from enum import Enum


class AbstractAgent(ABC):
    _ids = count(0)

    def __init__(self,pos,tp):
        self.name = 'Agent'
        self.id = next(self._ids)
        if pos[0]<0 or pos[1]<0:
            raise Exception("Positions going beyond edges.")
        self.tp = tp


    @abstractmethod
    def respond(self,observation):
        """
        Method to respond to an observation with a probability distributions on actions.
        :param observation: An observation sent from the environment
        :return proposal (action_probs,action_idx): A probability distribution over action space.
        """
        raise NotImplementedError

    @abstractmethod
    def act(self,proposal, decision: int):
        """
         Method to respond to an decision recvd from the environment based on a proposal the agent sent.
        :param proposal (action_probs,action_idx): The proposal that was sent by the agent.
        :param decision: The index of the action to perform recieved from the environment.
        :return:
        """
        raise NotImplementedError

    """
    @abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abstractmethod
    def __getstate__(self):
        raise NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        raise NotImplementedError
    """

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError



class AgentType():
    "Supporting class that holds a agent's type. This will help easy sharing/comparing/constructing of new types"

    status = Enum('Status', [('done', True), ('pending', False)])

    def __init__(self,stations):
        # If stations is an integer, randomly create station order to visit
        if isinstance(stations, int):
            self.n_stations = stations
            self.station_order = np.random.permutation(self.n_stations) #The order in which stations would be worked on.
            self.station_work_status = np.array([AgentType.status.pending]*self.n_stations) #The status of work on these stations. Everytime a station is worked on, it's work_status will be converted to True.
        # If stations is a list of integers (station indices), set station order to given list
        elif isinstance(stations, list):
            assert len(stations) == gd.N_STATIONS
            for stn in stations:
                # Elements of list must be station indices
                assert isinstance(stn, int)
                assert stn in range(gd.N_STATIONS)

            self.n_stations = len(stations)
            self.station_order = np.array(stations)
            self.station_work_status = np.array([AgentType.status.pending]*self.n_stations)


    def get_status(self):
        return copy.copy(self.station_work_status)

    def get_current_job_station(self):
        """
        Get the next free station to work on.
        """
        next_station = 0
        stn_idx = 0
        while(self.station_work_status[stn_idx] is AgentType.status.done):
            stn_idx += 1
        return self.station_order[stn_idx]

    def set_status(self,latest_station_id):
        """
        Adjust status of station's work.
        latest_station_id describes the latest station marked as done. This is the index of the station in the station_order vector. This method then marks it as True (done)
        """
        self.station_work_status[latest_station_id] = AgentType.status.done
        return True


    def __str__(self):
        stg = ''
        for sttn,status in zip(self.station_order,self.station_work_status):
            stg += str(sttn)
            if status == AgentType.status.done:
                stg += '*'
        return stg

    def __copy__(self):
        new_agent_type = agent_type(self.n_stations)
        new_agent_type.station_order = copy.deepcopy(self.station_order)
        new_agent_type.station_work_status = copy.deepcopy(self.station_work_status)
        return new_agent_type

    def __deepcopy__(self):
        return self.__copy__()

    def __eq__(self,new_tp):
        res = True
        res = res and (self.n_stations == new_tp.n_stations)
        res = res and (np.all(self.station_order == new_tp.station_order))
        res = res and (np.all(self.station_work_status == new_tp.station_work_status))
        return res
