import gym
from enum import IntEnum
import numpy as np


class ToolFetchingEnvironment(gym.Env):
    FETCHER_ACTIONS = IntEnum('FETCHER_Actions', 'RIGHT LEFT UP DOWN NOOP QUERY PICKUP', start=0)
    FETCHER_ACTION_VALUES = set(a.value for a in FETCHER_ACTIONS)
    WORKER_ACTIONS = IntEnum('WORKER_Actions', 'RIGHT LEFT UP DOWN NOOP WORK', start=0)
    WORKER_ACTION_VALUES = set(a.value for a in WORKER_ACTIONS)
    def __init__(self, fetcher_pos, worker_pos, stn_pos, tool_pos, worker_goal, width=10, height=10):
        assert len(stn_pos) == len(tool_pos)
        assert worker_goal >= 0 and worker_goal < len(stn_pos)
        self.width = width
        self.height = height
        self.f_pos = fetcher_pos
        self.w_pos = worker_pos
        self.s_pos = stn_pos
        self.t_pos = tool_pos
        self.curr_f_pos = fetcher_pos
        self.curr_w_pos = worker_pos
        self.curr_t_pos = tool_pos
        self.f_tool = None
        self.w_goal = worker_goal

    def make_fetcher_obs(self, w_action, f_action, answer=None):
        return (self.curr_w_pos, self.curr_f_pos, self.s_pos, self.curr_t_pos, w_action, f_action, answer)

    def make_worker_obs(self, w_action, f_action):
        return (self.curr_w_pos, self.curr_f_pos, self.s_pos, self.curr_t_pos, w_action, f_action, self.w_goal)


    def step(self, action_n):
        worker_action = action_n[0]
        fetcher_action, fetcher_details = action_n[1]
        assert worker_action in ToolFetchingEnvironment.WORKER_ACTION_VALUES
        assert fetcher_action in ToolFetchingEnvironment.FETCHER_ACTION_VALUES
        if fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY:
            answer = self.answer_query(fetcher_details)
            obs_n = np.array([self.make_worker_obs(ToolFetchingEnvironment.WORKER_ACTIONS.NOOP, ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY), self.make_fetcher_obs(ToolFetchingEnvironment.WORKER_ACTIONS.NOOP, ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY, answer)])
            reward_n = np.array([-1,-1])
            done_n = np.array([False, False])
            info_n = np.array([{}, {}])
            return obs_n, reward_n, done_n, info_n


        if worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT:
            self.curr_w_pos += np.array([1,0])
            self.curr_w_pos[0] = min(self.w_pos[0], self.width-1)
        elif worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.LEFT:
            self.curr_w_pos -= np.array([1,0])
            self.w_pos[0] = max(self.w_pos[0], 0)
        elif worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.UP:
            self.curr_w_pos -= np.array([0,1])
            self.curr_w_pos[1] = max(self.w_pos[1], 0)
        elif worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.DOWN:
            self.curr_w_pos += np.array([0,1])
            self.curr_w_pos[1] = min(self.w_pos[1], self.height-1)
        elif worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            goal_pos = self.s_pos[self.w_goal]
            tool_pos = self.curr_t_pos[self.w_goal]
            if self.curr_w_pos == goal_pos and self.curr_w_pos == tool_pos:
                obs_n = np.array([self.make_worker_obs(ToolFetchingEnvironment.WORKER_ACTIONS.WORK, fetcher_action), self.make_fetcher_obs(ToolFetchingEnvironment.WORKER_ACTIONS.WORK, fetcher_action)])
                reward_n = np.array([0,0])
                done_n = np.array([True, True])
                info_n = np.array([{}, {}])
                return obs_n, reward_n, done_n, info_n



        if fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT:
            self.curr_f_pos += np.array([1,0])
            self.curr_f_pos[0] = min(self.f_pos[0], self.width-1)
        elif fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT:
            self.curr_f_pos -= np.array([1,0])
            self.curr_f_pos[0] = max(self.f_pos[0], 0)
        elif fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.UP:
            self.curr_f_pos -= np.array([0,1])
            self.curr_f_pos[1] = max(self.f_pos[1], 0)
        elif fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN:
            self.curr_f_pos += np.array([0,1])
            self.curr_f_pos[1] = min(self.f_pos[1], self.height-1)
        elif fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP:
            assert fetcher_details >= 0 and fetcher_details < len(tool_pos)
            if self.curr_t_pos[fetcher_details] == self.curr_f_pos:
                self.f_tool = fetcher_details

        if self.f_tool:
            self.curr_t_pos[self.f_tool] = self.curr_f_pos


        obs_n = np.array([self.make_worker_obs(worker_action, fetcher_action), self.make_fetcher_obs(worker_action, fetcher_action)])
        reward_n = np.array([-1,-1])
        done_n = np.array([False, False])
        info_n = np.array([{}, {}])
        return obs_n, reward_n, done_n, info_n


    def answer_query(self, query):
        if self.w_goal in query:
            return True
        return False

    def reset(self):
        self.curr_f_pos = self.f_pos
        self.curr_w_pos = self.w_pos
        self.curr_t_pos = self.t_pos
        self.f_tool = None
        obs_n = np.array([self.make_worker_obs(ToolFetchingEnvironment.WORKER_ACTIONS.NOOP, ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP), self.make_fetcher_obs(ToolFetchingEnvironment.WORKER_ACTIONS.NOOP, ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP)])
        return obs_n
