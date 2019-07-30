import unittest
import pdb
import time
import ipdb
from src import environment
from src.agents import agent_lifter
from src.agents import agent_adhoc
import random
import numpy as np
from src import global_defs
from src import utils

n_tests=0

obj_pos, agent_pos = utils.generate_initial_conditions(global_defs.N_TYPES, 2)
i=0
# ipdb.set_trace()
if True:
    n_tests += 1
    print("-----------Test Iter: {}-------------".format(i))
    a1 = agent_lifter.agent_lifter(agent_pos[0], 2)
    a2 = agent_adhoc.agent_adhoc(agent_pos[1])

    env = environment.environment(global_defs.GRID_SIZE, obj_pos, True)

    env.register_agent(a1)
    env.register_adhoc_agent(a2)

    n_steps = 0
    # print("Target Distance: {}".format(target_dist))
    # print("Object location {}".format(object_location))
    while (not env.is_terminal):
        is_terminal, reward = env.step()
        # ipdb.set_trace()
        time.sleep(0.1)
        last_history = env.history[-1]
        n_steps += 1
        # print(n_steps)
        if n_steps==10:
            a1.tp = 4
        # if is_terminal:
    if is_terminal:
        print("Terminal Reward {}".format(reward))
        print("N_iters {}".format(n_steps))