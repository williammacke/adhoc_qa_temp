from src import global_defs
from src.global_defs import Point2D
from src.environment import environment
from src.agents.agent_leader import agent_leader
from src.agents.agent_adhoc_q import agent_adhoc

size = global_defs.GRID_SIZE
stn_pos = [(1,2), (0,4), (4,4)]
env = environment(size, stn_pos)

l_pos = Point2D(2, 3)
leader = agent_leader(l_pos)

a_pos = Point2D(2, 2)
a_tp = 1 # Not sure why type is a init parameter for adhoc and not leader. Type not even used in adhoc.__init__()
adhoc = agent_adhoc(a_pos, a_tp)
adhoc.register_tracking_agent(leader)

env.register_agent(leader)
env.register_adhoc_agent(adhoc)

# iteration = 0
# while(env.check_for_termination() and iteration < global_defs.MAX_ITERS):
#     env.step()
#     iteration += 1


# TODO:
# A lot of things:
#   haven't checked environment that much yet
#   agent_adhoc.act() is incomplete
#   in agent_leader, Agent_state namedtuple not used?
