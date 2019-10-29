import matplotlib
matplotlib.use('Agg') # Prevents error when showing plot over ssh
import matplotlib.pyplot as plt

from src import global_defs as gd
from src.global_defs import Point2D
from src.environment import environment
from src.agents import agent
from src.agents.agent_leader import agent_leader
from src.agents.agent_adhoc_q import agent_adhoc
from itertools import permutations

# Runs a single simulation
def experiment(size, stn_pos, tools_pos, l_pos, a_pos, l_tp=None, l_path=[], communication_timesteps=[], debug=False):
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


    env = environment(size, stn_pos, tools_pos)

    leader = agent_leader(l_pos, l_tp, l_path)

    adhoc = agent_adhoc(a_pos)
    adhoc.register_tracking_agent(leader)

    env.register_agent(leader)
    env.register_adhoc_agent(adhoc)

    env.register_communication_time_steps(communication_timesteps)

    step_count = 0
    terminated = False

    if debug:
        debug_output(step_count)

    while(not terminated and step_count < gd.MAX_ITERS):
        terminated, reward = env.step()

        step_count = env.step_count

        if debug:
            debug_output(step_count)

    return step_count - 1 # Not including last time step that agents are doing Action.WORK on station


# Returns a permutation of all unique optimal paths from leader agent to target station
def opt_path_perm(stn_pos, l_pos, l_station_order):
    l_path = []
    offset = stn_pos[l_station_order[0]] - l_pos

    if offset.x >= 0:
        l_path += [gd.Actions.RIGHT] * offset.x
    else:
        l_path += [gd.Actions.LEFT] * -offset.x
    if offset.y >= 0:
        l_path += [gd.Actions.UP] * offset.y
    else:
        l_path += [gd.Actions.DOWN] * -offset.y

    return set(permutations(l_path))


# Returns lists of timesteps to complete simulation for each query timestep
def get_query_timesteps(size, stn_pos, tools_pos, l_pos, a_pos, l_tp, all_leader_paths, debug=False):
    max_query = 1
    query = 0

    query_timesteps = []
    while query <= max_query:
        timesteps = []
        for path in all_leader_paths:
            steps = experiment(size, stn_pos, tools_pos, l_pos, a_pos, l_tp, list(path), [query], debug=debug)
            timesteps.append(steps)

        query_timesteps.append(timesteps)

        if query == 0:
            max_query = max(timesteps)
        query += 1

    return query_timesteps


size = 10
tools_pos = [Point2D(2,3)] # tools_pos needs to be an array but only one tool box is supported so far

l_pos = Point2D(5, 0)
# l_tp = agent.AgentType(len(stn_pos)) # Optional random order of stations to pass to agen_leader()
l_station_order = [2, 0, 1]
l_tp = agent.AgentType(l_station_order) # Optional fixed order of stations to pass to agent_leader()

a_pos = Point2D(4, 0)

# communication_timesteps = [] # list of time steps that communication occurs

# print('Number of path permutations:', len(all_leader_paths))
# print('Starting experiment...')

stn_pos = [Point2D(7,3), Point2D(3,8), Point2D(7,8)] # target station needs to be last listed if you want worst case scenario with wrong inferencing
all_leader_paths = opt_path_perm(stn_pos, l_pos, l_station_order)
query_timesteps1 = get_query_timesteps(size, stn_pos, tools_pos, l_pos, a_pos, l_tp, all_leader_paths)

stn_pos = [Point2D(3,8), Point2D(7,8), Point2D(7,3)]
all_leader_paths = opt_path_perm(stn_pos, l_pos, l_station_order)
query_timesteps2 = get_query_timesteps(size, stn_pos, tools_pos, l_pos, a_pos, l_tp, all_leader_paths)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

ax[0].boxplot(query_timesteps1, positions=range(len(query_timesteps1)), whis='range')
ax[0].set_title('Timestep Range Based on Query Times: Station 2')
ax[0].set_xlabel('Query Timestep')
ax[0].set_ylabel('Timesteps')

ax[1].boxplot(query_timesteps2, positions=range(len(query_timesteps2)), whis='range')
ax[1].set_title('Timestep Range Based on Query Times: Station 3')
ax[1].set_xlabel('Query Timestep')
ax[1].set_ylabel('Timesteps')

# plt.show()
plt.savefig('testgraph')
