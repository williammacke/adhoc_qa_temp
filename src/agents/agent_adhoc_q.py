"""
This file contains relevant code for Fetcher's Policies
"""
from src.agents.agent import Policy
from src.environment import ToolFetchingEnvironment
import numpy as np
import random
from scipy.optimize import fsolve
from statistics import  median


# Returns list of valid actions that brings fetcher closer to all tools
def get_valid_actions(obs, agent):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    valid_actions = np.array([True] * 4) # NOOP is always valid
    for stn in range(len(s_pos)):
        if agent.probs[stn] == 0:
            continue
        tool_valid_actions = np.array([True] * 4)
        if f_pos[0] <= t_pos[stn][0]:
            tool_valid_actions[1] = False # Left
        elif f_pos[0] >= t_pos[stn][0]:
            tool_valid_actions[0] = False # Right
        if f_pos[1] >= t_pos[stn][1]:
            tool_valid_actions[2] = False # Down
        elif f_pos[1] <= t_pos[stn][1]:
            tool_valid_actions[3] = False # Up

        valid_actions = np.logical_and(valid_actions, tool_valid_actions)

    return valid_actions


def never_query(obs, agent):
    return None


def random_query(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None
    possible_stations = [s for s, s_p in enumerate(agent.probs) if s_p > 0]
    return random.sample(possible_stations, len(possible_stations) // 2)


def max_action_query(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    s_probs = agent.probs

    stn_per_action = {
            ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.UP: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN: []
    }
    for i, t in enumerate(t_pos):
        if s_probs[i] == 0:
            continue
        if f_pos[0] < t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT].append(i)
        if f_pos[0] > t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT].append(i)
        if f_pos[1] < t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.UP].append(i)
        if f_pos[1] > t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN].append(i)

    return max(stn_per_action.values(), key=len)

def min_action_query(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    s_probs = agent.probs

    stn_per_action = {
            ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.UP: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN: []
    }
    for i, t in enumerate(t_pos):
        if s_probs[i] == 0:
            continue
        if f_pos[0] < t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT].append(i)
        if f_pos[0] > t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT].append(i)
        if f_pos[1] < t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.UP].append(i)
        if f_pos[1] > t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN].append(i)

    return min(stn_per_action.values(), key=len)

def median_action_query(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    s_probs = agent.probs

    stn_per_action = {
            ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.UP: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN: []
    }
    for i, t in enumerate(t_pos):
        if s_probs[i] == 0:
            continue
        if f_pos[0] < t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT].append(i)
        if f_pos[0] > t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT].append(i)
        if f_pos[1] < t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.UP].append(i)
        if f_pos[1] > t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN].append(i)
    stns_per_action_values = list(stn_per_action.values())
    stns_per_action_values.sort(key=len)

    return stns_per_action_values[len(stns_per_action_values)//2]


class FetcherQueryPolicy(Policy):
    """
    Basic Fetcher Policy for querying, follows query_policy function argument (defaults to never query)
    Assumes all tools are in same location
    """
    def __init__(self, query_policy=never_query):
        self.query_policy = query_policy
        self.probs = None
        self.query = None
        self.prev_w_pos = None


    def reset(self):
        self.probs = None
        self.query = None
        self.prev_w_pos = None


    def make_inference(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.prev_w_pos is None:
            return
        if w_action == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            for i,stn in enumerate(s_pos):
                if not np.array_equal(stn, self.prev_w_pos):
                    self.probs[i] = 0
        elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT:
            for i,stn in enumerate(s_pos):
                if stn[0] <= self.prev_w_pos[0]:
                    self.probs[i] = 0
        elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.LEFT:
            for i,stn in enumerate(s_pos):
                if stn[0] >= self.prev_w_pos[0]:
                    self.probs[i] = 0
        elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.DOWN:
            for i,stn in enumerate(s_pos):
                if stn[1] >= self.prev_w_pos[1]:
                    self.probs[i] = 0
        elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.UP:
            for i,stn in enumerate(s_pos):
                if stn[1] <= self.prev_w_pos[1]:
                    self.probs[i] = 0

        self.probs /= np.sum(self.probs)


    def action_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT)
        elif pos[0] > goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT)
        if pos[1] > goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN)
        elif pos[1] < goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.UP)
        if len(actions) == 0:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP
        return np.random.choice(actions)


    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
            else:
                for stn in self.query:
                    self.probs[stn] = 0
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY, self.query

        if np.max(self.probs) < 1:
            #dealing with only one tool position currently
            if np.array_equal(f_pos, t_pos[0]):
                return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP, None
            else:
                return self.action_to_goal(f_pos, t_pos[0]), None
        else:
            if f_tool != np.argmax(self.probs):
                if np.array_equal(f_pos, t_pos[0]):
                    return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, np.argmax(self.probs)
                else:
                    return self.action_to_goal(f_pos, t_pos[0]), None
            return self.action_to_goal(f_pos, s_pos[np.argmax(self.probs)]), None


class FetcherAltPolicy(FetcherQueryPolicy):
    """
    More Complicated Fetcher Policy, allows for multiple tool locations
    """
    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
            else:
                for stn in self.query:
                    self.probs[stn] = 0
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        # One station already guaranteed. No querying needed.
        if np.max(self.probs) == 1:
            target = np.argmax(self.probs)
            if f_tool != target:
                if np.array_equal(f_pos, t_pos[target]):
                    return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, target
                else:
                    return self.action_to_goal(f_pos, t_pos[target]), None
            return self.action_to_goal(f_pos, s_pos[target]), None

        self.query = self.query_policy(obs, self)
        if self.query:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY, self.query

        valid_actions = get_valid_actions(obs, self)

        if np.any(valid_actions):
            p = valid_actions / np.sum(valid_actions)
            action_idx = np.random.choice(np.arange(4), p=p)
            return ToolFetchingEnvironment.FETCHER_ACTIONS(action_idx), None
        else:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP, None

class FetcherAgentTypePolicy(Policy):
    def __init__(self, agent_classifier, query_policy=never_query):
        self._query_policy = query_policy
        self._agent_classifier = agent_classifier
        self._probs = None
        self._full_probs = None

    def reset(self):
        self._agent_classifier.reset()
        self._probs = None
        self._full_probs = None

    def make_inference(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        num_g = len(s_pos)
        num_a = self._agent_classifier.num_agent_types
        if not self._agent_classifier.initialized:
            self._agent_classifier.init(obs)
        if self._full_probs is None:
            self._full_probs = np.ones((num_g, num_a))
            self._full_probs /= np.sum(self._full_probs)
        if self._probs is None:
            self._probs = np.empty(num_g)
        self._full_probs *= self._agent_classifier(obs)
        self._full_probs /= np.sum(self._full_probs)

        for i in range(num_g):
            self._probs[i] = np.sum(self._full_probs[i, :])



    def _action_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT)
        elif pos[0] > goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT)
        if pos[1] > goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN)
        elif pos[1] < goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.UP)
        if len(actions) == 0:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP
        return np.random.choice(actions)


    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if answer is not None:
            raise NotImplementedError
        else:
            self.make_inference(obs)
        goal = np.argmax(self._probs)
        if f_tool == goal:
            return self._action_to_goal(f_pos, s_pos[goal]), None
        else:
            if np.array_equal(f_pos, t_pos[goal]):
                return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, goal
            else:
                return self._action_to_goal(f_pos, t_pos[goal]), None
