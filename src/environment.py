import gym
from enum import IntEnum


class ToolFetchingEnvironment(gym.Env):
    FETCHER_ACTIONS = IntEnum('FETCHER_Actions', 'RIGHT LEFT UP DOWN NOOP WORK QUERY', start=0)
    FETCHER_ACTION_VALUES = set(a.value for a in FETCHER_ACTIONS)
    WORKER_ACTIONS = IntEnum('WORKER_Actions', 'RIGHT LEFT UP DOWN NOOP', start=0)
    WORKER_ACTION_VALUES = set(a.value for a in WORKER_ACTIONS)
    def __init__(self, fetcher_pos, worker_pos, stn_pos, tool_pos, width=10, height=10):
        assert len(stn_pos) == len(tool_pos)
        self.width = width
        self.height = height
        self.f_pos = fetcher_pos
        self.w_pos = worker_pos
        self.s_pos = stn_pos
        self.t_pos = tool_pos
        self.curr_f_pos = fetcher_pos
        self.curr_w_pos = worker_pos
        self.curr_t_pos = tool_pos

    def makeObs(self):
        return (self.curr_w_pos, self.curr_f_pos, self.s_pos, self.curr_t_pos)

    def step(self, action_n):
        worker_action = action_n[0]
        fetcher_action = action_n[1]
        assert worker_action in ToolFetchingEnvironment.WORKER_ACTION_VALUES
        assert fetcher_action in ToolFetchingEnvironment.FETCHER_ACTION_VALUES
        raise NotImplementedError

    def reset(self):
        self.curr_f_pos = self.f_pos
        self.curr_w_pos = self.w_pos
        self.curr_t_pos = self.t_pos
        return [self.makeObs()]*2
