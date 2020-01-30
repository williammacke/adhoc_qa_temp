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

        if w_pos[1] < goal_pos[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.DOWN)
        elif w_pos[1] > goal_pos[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.UP)

        return np.random.choice(actions)
