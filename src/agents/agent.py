"""
Defines abstract base clsas of what an Agent should be like. It can get more and more complicated as time passes, with complications inside agent models.
"""
from abc import ABC, abstractmethod, abstractproperty
from .. import global_defs
import numpy as np
from itertools import count


class Agent(ABC):
    _ids = count(0)

    def __init__(self,pos,tp):
        self.name = 'Agent'
        self.id = next(self._ids)
        if pos[0]<0 or pos[1]<0:
            raise Exception("Positions going beyond edges.")


    @abstractmethod
    def respond(self,observation):
        """
        Method to respond to an observation with a probability distributions on actions.
        :param observation: An observation sent from the environment
        :return action_probs: A probability distribution over action space.
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

    @abstractmethod
    def __copy__(self):
        raise NotImplementedError

    @abstractmethod
    def __getstate__(self):
        raise NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError




