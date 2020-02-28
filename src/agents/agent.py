import numpy as np
from src.environment import ToolFetchingEnvironment

class Policy:
    def __call__(self, obs):
        raise NotImplementedError

class RandomWorkerPolicy(Policy):
    def __init__(self):
        self.last_w_pos = None
    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, goal = obs
        goal_pos = s_pos[goal]
        if np.array_equal(goal_pos, w_pos):
            return ToolFetchingEnvironment.WORKER_ACTIONS.WORK
        actions = []
        if w_pos[0] < goal_pos[0]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT)
        elif w_pos[0] > goal_pos[0]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.LEFT)

        if w_pos[1] > goal_pos[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.DOWN)
        elif w_pos[1] < goal_pos[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.UP)

        return np.random.choice(actions)

class PlanPolicy(Policy):
    def __init__(self, plan):
        self._plan = plan
        self._step = 0

    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, goal = obs
        if f_action != ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY:
            action = self._plan[min(self._step, len(self._plan)-1)]
            self._step += 1
            return action
        return self._plan[min(self._step-1, len(self._plan)-1)]

    def reset(self):
        self._step = 0

    @property
    def step(self):
        return self._step

    @property
    def plan(self):
        return self._plan
