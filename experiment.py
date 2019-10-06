from src import global_defs as gd
from src.global_defs import Point2D
from src.environment import environment
from src.agents import agent
from src.agents.agent_leader import agent_leader
from src.agents.agent_adhoc_q import agent_adhoc

size = 10
stn_pos = [Point2D(1,2), Point2D(0,4), Point2D(4,4)]
# Need to implement tool box
env = environment(size, stn_pos)

l_pos = Point2D(2, 3)
# l_tp = agent.AgentType([1,0,2]) # Optional fixed order of stations to pass to agent_leader()
leader = agent_leader(l_pos)

a_pos = Point2D(2, 2)
adhoc = agent_adhoc(a_pos)
adhoc.register_tracking_agent(leader)

env.register_agent(leader)
env.register_adhoc_agent(adhoc)

env.step()


# iteration = 0
# terminated = False
# while(not terminated and iteration < gd.MAX_ITERS):
#     terminated, reward = env.step()
#     iteration += 1


# TODO:
# A lot of things:
#   haven't checked environment that much yet
#   agent_adhoc.act() is incomplete
#   environment.check_for_termination() not completed
#   in agent_leader, Agent_state namedtuple not used?
