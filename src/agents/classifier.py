import numpy as np
from src.environment import ToolFetchingEnvironment

class Classifier:
    def __init__(self):
        self._initialized = False
        self._num_g = 0
        self._goals = None
        self._prev_w_pos = None

    def reset():
        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    def init(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        self._initialized = True
        self._num_g = len(s_pos)
        self._goals = s_pos
        self._prev_w_pos = None

    @property
    def num_goal_types(self):
        return self._num_g

    @property
    def num_agent_types(self):
        raise NotImplementedError

    @property
    def num_types(self):
        return self.num_goal_types*self.num_agent_types

    def __call__(self, obs):
        raise NotImplementedError


class EpsilonGreedyClassifier(Classifier):
    def __init__(self, epsilons):
        super().__init__()
        self._epsilons = epsilons

    @property
    def num_agent_types(self):
        return len(self._epsilons)

    def _actions_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT)
        elif pos[0] > goal[0]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.LEFT)
        if pos[1] > goal[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.DOWN)
        elif pos[1] < goal[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.UP)
        if len(actions) == 0:
            return set([ToolFetchingEnvironment.WORKER_ACTIONS.WORK])
        return set(actions)

    def __call__(self, obs):
        assert self.initialized
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self._prev_w_pos is None:
            self._prev_w_pos = np.array(w_pos)
            return np.ones(self.num_types)
        probs = np.empty(self.num_types)
        i = 0
        for g in self._goals:
            for e in self._epsilons:
                valid_actions = self._actions_to_goal(self._prev_w_pos, g)
                if w_action in valid_actions:
                    probs[i] = 1-e
                else:
                    probs[i] = e
                i += 1
        self._prev_w_pos = np.array(w_pos)
        return probs
