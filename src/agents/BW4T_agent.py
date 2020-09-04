import numpy as np
from src.environment2 import BlockWorld4Teams


class Agent:
    def answer(self, query):
        raise NotImplementedError

    def __call__(self, state):
        raise NotImplementedError

    def receive(self, query, answer):
        raise NotImplementedError

    def reset(self, state):
        raise NotImplementedError


def action_to_goal(pos, goal):
    if pos < goal:
        return BlockWorld4Teams.ACTIONS.RIGHT, None
    elif pos > goal:
        return BlockWorld4Teams.ACTIONS.LEFT, None
    else:
        return BlockWorld4Teams.ACTIONS.NOOP, None


class GreedyAgent(Agent):
    def __init__(self):
        pass

    def __call__(self, state, a_index, drop_room):
        a_pos, b_pos, ab, order = state
        current_block = len(np.where(b_pos == -1)[0])
        while current_block in ab:
            if np.where(ab == current_block)[0] == a_index:
                break
            current_block += 1
        if current_block >= len(order):
            return BlockWorld4Teams.ACTIONS.NOOP, None
        target = order[current_block]
        loc = a_pos[a_index]
        t_loc = b_pos[target]
        if target == ab[a_index]:
            return action_to_goal(loc, drop_room)
        else:
            if loc == t_loc:
                return BlockWorld4Teams.ACTIONS.PICKUP, target
            return action_to_goal(loc, t_loc)

    def receive(self, query, answer):
        pass

    def reset(self, state):
        pass

    def answer(self, state, query):
        a_pos, b_pos, ab, order = state
        current_block = len(np.where(b_pos == -1)[0])
        target = order[current_block]
        return target in query

        


