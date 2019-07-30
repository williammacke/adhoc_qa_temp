import numpy as np
import random
import ipdb
import time

from src.agents import agent_lifter
from src import environment
from src import global_defs

# Agents: Same type.
# Object: Not on edges.
# Distance: target_dist steps.

object_location = np.array(random.sample(range(1, 8), 2)[:2])

# Put agents target_dist positions away.
target_dist = 6
displacement = np.random.randint(-target_dist, target_dist, (2, 2))
displacement[:, 1] = random.choice([-1, 1]) * (target_dist - np.abs(displacement[:, 0]))

a1_pos = (object_location[0] + displacement[0][0], object_location[1] + displacement[0][1])
a2_pos = (object_location[0] + displacement[1][0], object_location[1] + displacement[1][1])


a1 = agent_lifter.agent_lifter(a1_pos, 1)
a2 = agent_lifter.agent_lifter(a2_pos, 1)

object_location = global_defs.Point2D(object_location[0], object_location[1])

env = environment.environment(10, [object_location, global_defs.Point2D(0, 0)], True)
env.register_agent(a1)
env.register_agent(a2)

n_steps = 0
print("A1 {}".format(a1.pos))
print("A2 {}".format(a2.pos))

while (not env.is_terminal and n_steps < 10):

    is_terminal, reward = env.step()

    n_steps += 1
    print(n_steps)
    print("A1 {}".format(a1.pos))
    print("A2 {}".format(a2.pos))
    time.sleep(1)

if env.is_terminal:
    print("Terminal Reward {}".format(reward))
    print("N_iters {}".format(n_steps))
# print("n_tests {} total_tests {}".format(n_tests,total_test))