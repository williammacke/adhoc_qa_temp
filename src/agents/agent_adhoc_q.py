"""
This file contains relevant code for Fetcher's Policies
"""
from src.agents.agent import Policy
from src.environment import ToolFetchingEnvironment
import numpy as np
import random
from scipy.optimize import fsolve
from statistics import  median
import pulp
import copy


def is_ZB(obs, g1, g2):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    valid_actions1 = np.array([True] * 4)
    if f_pos[0] <= t_pos[g1][0]:
        valid_actions1[1] = False # Left
    elif f_pos[0] >= t_pos[g1][0]:
        valid_actions1[0] = False # Right
    if f_pos[1] >= t_pos[g1][1]:
        valid_actions1[2] = False # Down
    elif f_pos[1] <= t_pos[g1][1]:
        valid_actions1[3] = False # Up

    valid_actions2 = np.array([True] * 4)
    if f_pos[0] <= t_pos[g2][0]:
        valid_actions2[1] = False # Left
    elif f_pos[0] >= t_pos[g2][0]:
        valid_actions2[0] = False # Right
    if f_pos[1] >= t_pos[g2][1]:
        valid_actions2[2] = False # Down
    elif f_pos[1] <= t_pos[g2][1]:
        valid_actions2[3] = False # Up

    return not np.any(np.logical_and(valid_actions1, valid_actions2))


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
        if f_pos[0] >= t_pos[stn][0]:
            tool_valid_actions[0] = False # Right
        if f_pos[1] >= t_pos[stn][1]:
            tool_valid_actions[2] = False # Down
        if f_pos[1] <= t_pos[stn][1]:
            tool_valid_actions[3] = False # Up

        valid_actions = np.logical_and(valid_actions, tool_valid_actions)

    return valid_actions


def never_query(obs, agent):
    return None


def random_query(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None
    possible_stations = [s for s, s_p in enumerate(agent.probs) if s_p > 0]
    query =  random.sample(possible_stations, len(possible_stations) // 2)
    #print(query)
    return query


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
    valid_queries = {k:v for k,v in stn_per_action.items() if len(v) > 0}

    return min(valid_queries.values(), key=len)

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

def smart_query(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split", pulp.LpMaximize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum(alpha[i,j] - x[i] - x[j] for i,j in zbs)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    problem.solve()

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    query =  list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(query) == 0:
        return None
    return query


def smart_query2(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split", pulp.LpMinimize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum((1 + x[i] + x[j] - alpha[i,j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    problem.solve()

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    query = list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(query) == 0:
        return None
    return query


def create_smart_query3(cost):
    def smart_query3(obs, agent):
        if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
            return None
    
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    
        zbs = set()
        goals = []
        zb_goals = set()
        for g in range(len(s_pos)):
            if agent.probs[g] == 0:
                continue
            goals.append(g)
    
        for g1 in goals:
            for g2 in goals:
                if g1 == g2:
                    continue
                if (g2, g1) in zbs:
                    continue
                if is_ZB(obs, g1, g2):
                    zbs.add((g1, g2))
                    zb_goals.add(g1)
                    zb_goals.add(g2)
    
        bin1 = set()
        bin2 = set()
        used = set()
    
        problem = pulp.LpProblem("Find max split with cost", pulp.LpMinimize)
    
        x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
        alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
        problem += pulp.lpSum((1 + x[i] + x[j] - alpha[i,j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs) + pulp.lpSum(cost * x[i] for i in x)
        for i,j in zbs:
            problem += alpha[i,j] <= 2*(x[i]+x[j])
        problem.solve()
    
        for i in x:
            if x[i].varValue == 1:
                bin1.add(i)
            elif x[i].varValue == 0:
                bin2.add(i)
            else:
                raise ValueError
    
        remaining = []
        for g in goals:
            if g in zb_goals:
                continue
            remaining.append(g)
        query =  list(bin1) + random.sample(remaining, len(remaining)//2)
        if len(query) == 0:
            return None
        return query
    return smart_query3

def smart_query3(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split with cost", pulp.LpMinimize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum((1 + x[i] + x[j] - alpha[i,j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs) + pulp.lpSum(0.1 * x[i] for i in x)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    problem.solve()

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    query =  list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(query) == 0:
        return None
    return query


def smart_query_noRandom(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split", pulp.LpMaximize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum(alpha[i,j] - x[i] - x[j] for i,j in zbs)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    problem.solve()

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    #return list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(bin1) == 0:
        return None
    return list(bin1)


def smart_query2_noRandom(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split", pulp.LpMinimize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum((1 + x[i] + x[j] - alpha[i,j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    problem.solve()

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    #return list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(bin1) == 0:
        return None

    return list(bin1)


def create_smart_query3_noRandom(cost):
    def smart_query3_noRandom(obs, agent):
        if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
            return None
    
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    
        zbs = set()
        goals = []
        zb_goals = set()
        for g in range(len(s_pos)):
            if agent.probs[g] == 0:
                continue
            goals.append(g)
    
        for g1 in goals:
            for g2 in goals:
                if g1 == g2:
                    continue
                if (g2, g1) in zbs:
                    continue
                if is_ZB(obs, g1, g2):
                    zbs.add((g1, g2))
                    zb_goals.add(g1)
                    zb_goals.add(g2)
    
        bin1 = set()
        bin2 = set()
        used = set()
    
        problem = pulp.LpProblem("Find max split with cost", pulp.LpMinimize)
    
        x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
        alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
        problem += pulp.lpSum((1 + x[i] + x[j] - alpha[i,j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs) + pulp.lpSum(cost * x[i] for i in x)
        for i,j in zbs:
            problem += alpha[i,j] <= 2*(x[i]+x[j])
        problem.solve()
    
        for i in x:
            if x[i].varValue == 1:
                bin1.add(i)
            elif x[i].varValue == 0:
                bin2.add(i)
            else:
                raise ValueError
    
        remaining = []
        for g in goals:
            if g in zb_goals:
                continue
            remaining.append(g)
        #return list(bin1) + random.sample(remaining, len(remaining)//2)
        if len(bin1) == 0:
            return None
        return list(bin1)
    return smart_query3_noRandom


def smart_query3_noRandom(obs, agent):
    if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split with cost", pulp.LpMinimize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum((1 + x[i] + x[j] - alpha[i,j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs) + pulp.lpSum(0.1 * x[i] for i in x)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    problem.solve()

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    #return list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(bin1) == 0:
        return None
    return list(bin1)
    

    """

    for g1, g2 in zbs:
        if g1 in used and g2 in used:
            continue
        if g1 in used:
            if g1 in bin1:
                bin2.add(g2)
            else:
                bin1.add(g2)
        elif g2 in used:
            if g2 in bin1:
                bin2.add(g1)
            else:
                bin1.add(g1)
        else:
            bin1.add(g1)
            bin2.add(g2)
        used.add(g1)
        used.add(g2)

    valid = []
    for g in goals:
        if g in used: continue
        valid.append(g)

    if len(bin1) < len(bin2):
        rbin = bin1
        obin = bin2
    else:
        rbin = bin2
        obin = bin1
        

    return list(rbin) + random.sample(valid, len(valid)//2) + random.sample(obin, (len(obin) - len(rbin))//2)
    #return list(bin1)
    """








class FetcherQueryPolicy(Policy):
    """
    Basic Fetcher Policy for querying, follows query_policy function argument (defaults to never query)
    Assumes all tools are in same location
    """
    def __init__(self, query_policy=never_query, prior=None, epsilon=0):
        self.query_policy = query_policy
        self._prior = prior
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None
        self._epsilon = epsilon


    def reset(self):
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None


    def make_inference(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.prev_w_pos is None:
            return
        if w_action == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            for i,stn in enumerate(s_pos):
                if not np.array_equal(stn, self.prev_w_pos):
                    self.probs[i] = self._epsilon
        elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT:
            for i,stn in enumerate(s_pos):
                if stn[0] <= self.prev_w_pos[0]:
                    self.probs[i] = self._epsilon
        elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.LEFT:
            for i,stn in enumerate(s_pos):
                if stn[0] >= self.prev_w_pos[0]:
                    self.probs[i] = self._epsilon
        elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.DOWN:
            for i,stn in enumerate(s_pos):
                if stn[1] >= self.prev_w_pos[1]:
                    self.probs[i] = self._epsilon
        elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.UP:
            for i,stn in enumerate(s_pos):
                if stn[1] <= self.prev_w_pos[1]:
                    self.probs[i] = self._epsilon

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

        if np.max(self.probs) < (1 - self._epsilon):
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
        if np.max(self.probs) >= (1 - self._epsilon):
            target = np.argmax(self.probs)
            if f_tool != target:
                if np.array_equal(f_pos, t_pos[target]):
                    return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, target
                else:
                    return self.action_to_goal(f_pos, t_pos[target]), None
            return self.action_to_goal(f_pos, s_pos[target]), None

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY, self.query

        valid_actions = get_valid_actions(obs, self)

        if np.any(valid_actions):
            #print(valid_actions)
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

        if not any(p > 0 for p in self._full_probs.flatten()):
            self._full_probs = np.ones((num_g, num_a))
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
