from src import global_defs as gd
from src.global_defs import Point2D
from src.environment import environment
from src.agents import agent
from src.agents.agent_leader import agent_leader
from src.agents.agent_adhoc_q import agent_adhoc


# DEBUG CODE
def debug_output(timestep=None):
    print('------------------------------------------\n')
    print('TIME STEP:            ', timestep)
    station_positions = [str(station) for station in stn_pos]
    print('\nStation Positions:    ', station_positions)
    toolbox_positions = [str(toolbox) for toolbox in tools_pos]
    print('Toolbox Position:     ', toolbox_positions)
    print('\nLeader Pos:           ', leader.pos)
    # print('Leader Target:        ', leader.tp.get_current_job_station()) # This throws index out of range error when all stations are complete
    leader_action = env.allActions[gd.LEADER_IDX]
    if leader_action is not None:
        leader_action = gd.Actions_list[leader_action]
    print('Leader Action:        ', leader_action)
    print('Leader Station Order: ', leader.tp.station_order)
    leader_station_status = [status.value for status in leader.tp.station_work_status]
    print('Leader Station Status:', leader_station_status)
    print('\nAdhoc Pos:            ', adhoc.pos)
    adhoc_action = env.allActions[gd.ADHOC_IDX]
    if adhoc_action is not None:
        adhoc_action = gd.Actions_list[adhoc_action]
    print('Adhoc Action:         ', adhoc_action)
    print('Adhoc Tool:           ', adhoc.tool)
    print('Adhoc Station Order:  ', adhoc.knowledge.station_order)
    print('Adhoc Source:         ', adhoc.knowledge.source)
    adhoc_station_status = [status.value for status in adhoc.knowledge.station_work_status]
    print('Adhoc Station Status: ', adhoc_station_status)
    print('\n')
    return


size = 10
stn_pos = [Point2D(7,3), Point2D(7,8), Point2D(3,8)]
tools_pos = [Point2D(2,3)] # tools_pos needs to be an array but only one tool box is supported so far
env = environment(size, stn_pos, tools_pos)

l_pos = Point2D(5, 0)
atype = agent.AgentType(len(stn_pos)) # Optional random order of stations to pass to agen_leader()
l_tp = agent.AgentType([1,0,2]) # Optional fixed order of stations to pass to agent_leader()
leader = agent_leader(l_pos, l_tp)

a_pos = Point2D(4, 0)
adhoc = agent_adhoc(a_pos)
adhoc.register_tracking_agent(leader)

env.register_agent(leader)
env.register_adhoc_agent(adhoc)

step_count = 0
terminated = False
debug_output(step_count)
while(not terminated and step_count < gd.MAX_ITERS):
    terminated, reward = env.step()

    step_count = env.step_count

    debug_output(step_count)
