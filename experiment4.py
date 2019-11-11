import matplotlib
#matplotlib.use('Agg') # Prevents error when 'showing' plot over ssh
import matplotlib.pyplot as plt
import math
import numpy as np

from src import global_defs as gd
from src.global_defs import Point2D
from src.environment import environment
from src.agents import agent
from src.agents.agent_leader import agent_leader
from src.agents.agent_adhoc_q import agent_adhoc
from itertools import permutations


def getQuerySets(goals):
    goals2 = np.array(goals)
    np.random.shuffle(goals2)
    return goals2[:len(goals2)//2]

# Runs a single simulation
def experiment(grid_size, stn_pos, tools_pos, l_pos, a_pos, l_tp=None, l_path=[], communication_timesteps=[], debug=False):

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


    env = environment(grid_size, stn_pos, tools_pos)

    leader = agent_leader(l_pos, l_tp, l_path)

    adhoc = agent_adhoc(a_pos)
    adhoc.register_tracking_agent(leader)

    env.register_agent(leader)
    env.register_adhoc_agent(adhoc)

    env.register_communication(communication_timesteps, getQuerySets)

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
def opt_path_perm(stn_pos, l_pos, l_station_order, repeat=1):
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

    #paths = list(set(permutations(l_path))) * repeat
    np.random.shuffle(l_path)
    paths = [l_path] * repeat

    return paths


# Returns lists of timesteps to complete simulation for each query timestep
def get_query_timesteps(grid_size, stn_pos, tools_pos, l_pos, a_pos, l_tp, all_leader_paths, num_query=None, debug=False):
    max_query = 0
    tx = tools_pos[0][0]
    ty = tools_pos[0][1]
    queries = [0,1,abs(tx-a_pos[0])+abs(ty-a_pos[1]), -1] 

    query_timesteps = []
    for query in queries:
        print(query)
        timesteps = []
        for path in all_leader_paths:
            if query == -1:
                qp = int(np.random.random()*(max_query-1))+1
            else:
                qp = query
            steps = experiment(grid_size, stn_pos, tools_pos, l_pos, a_pos, l_tp, list(path), list(range(qp, max_query+1)), debug=debug)
            timesteps.append(steps)

        if query == 0:
            max_query = max(timesteps)
        query_timesteps.append(timesteps)

    return query_timesteps


def create_graphs(grid_size, stn_pos_perm, stn_names, tools_pos, l_pos, a_pos, l_tp_perm, num_query=None):
    query_timesteps = [[] for _ in range(4)]

    for stn_pos, l_tp in zip(stn_pos_perm, l_tp_perm):
        tools_pos = [Point2D(int(np.random.random()*50), int(np.random.random()*50))]
        l_pos = Point2D(int(np.random.random()*50), int(np.random.random()*50))
        a_pos = Point2D(int(np.random.random()*50), int(np.random.random()*50))
        all_leader_paths = opt_path_perm(stn_pos, l_pos, l_tp.station_order, 10)
        qt = get_query_timesteps(grid_size, stn_pos, tools_pos, l_pos, a_pos, l_tp, all_leader_paths, num_query, debug=False)
        for i in range(4):
            query_timesteps[i] += qt[i]

    np.savetxt("query_until_certain.dat", query_timesteps)
    query_timesteps = [query_timesteps]


    num_graphs = len(query_timesteps)
    fig_width = 10
    fig_height = 3 

    fig, ax = plt.subplots(num_graphs, 1, figsize=(fig_width, fig_height))
    fig2, ax2 = plt.subplots(num_graphs, 1, figsize=(fig_width, fig_height))

    for i in range(len(query_timesteps)):
        positions = list(range(len(query_timesteps[i])))
        if num_query:
            positions = list(range(num_query))
        labels = positions.copy()
        labels[0] = 'X'
        labels[2] = 'Zq'
        labels[3] = 'random'

        if num_graphs > 1:
            axs = ax[i]
        else:
            axs = ax
        axs.boxplot(query_timesteps[i], positions=positions, whis='range', labels=labels)
        axs.set_title('Timestep Range Based on Query Times: Station %s' % (stn_names[i]))
        axs.set_xlabel('Query Timestep')
        axs.set_ylabel('Timesteps')



        avg = [sum(x) / len(x) for x in query_timesteps[i]]
        sd = []
        for d, mu in zip(query_timesteps[i], avg):
            s = [(x - mu)**2 for x in d]
            sd.append(math.sqrt(sum(s) / len(s)))

        if num_graphs > 1:
            axs2 = ax2[i]
        else:
            axs2 = ax2

        axs2.errorbar(positions, avg, sd)
        axs2.set_title('Timestep Average and Standard Deviation: Station %s' % (stn_names[i]))
        axs2.set_xlabel('Query Timestep')
        axs2.set_ylabel('Timesteps')

        print('Graph', i, ': ')
        print(avg)
        print(sd)

    fig.tight_layout()
    fig2.tight_layout()

    plt.show()
    # plt.savefig('testgraph')


grid_size = 50
# target station needs to be last listed if you want worst case scenario with wrong inferencing
#stn_names = ['1', '2', '3']
stn_names = [str(i) for i in range(1,11)]
#stn_pos_perm = [[Point2D(7,3), Point2D(7,8), Point2D(3,8)],
#                [Point2D(7,3), Point2D(3,8), Point2D(7,8)],
#                [Point2D(3,8), Point2D(7,8), Point2D(7,3)]]
nexp = 1000
pos = np.array([np.random.choice(2500,size=10,replace=False) for _ in range(nexp)])
stn_pos_perm = [[Point2D(pos[i][j]//50, pos[i][j]%50) for j in range(10)] for i in range(nexp)]
tools_pos = [Point2D(2,3)] # tools_pos needs to be an array but only one tool box is supported so far

l_pos = Point2D(5, 0)
# l_tp = agent.AgentType(len(stn_pos)) # Optional random order of stations to pass to agent_leader()
#l_tp_perm = [agent.AgentType([2]), agent.AgentType([2]), agent.AgentType([2])] # Optional fixed order of stations to pass to agent_leader()
l_tp_perm = [agent.AgentType([int(np.random.random()*10)]) for _ in range(nexp)] # Optional fixed order of stations to pass to agent_leader()

a_pos = Point2D(4, 0)

create_graphs(grid_size, stn_pos_perm, stn_names, tools_pos, l_pos, a_pos, l_tp_perm)

# experiment(grid_size, stn_pos_perm[0], tools_pos, l_pos, a_pos, l_tp_perm[0], [gd.Actions.LEFT], [3], debug=True)
