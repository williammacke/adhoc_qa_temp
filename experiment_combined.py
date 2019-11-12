import matplotlib
#matplotlib.use('Agg') # Prevents error when 'showing' plot over ssh
import matplotlib.pyplot as plt
import math

from src import global_defs as gd
from src.global_defs import Point2D
from src.environment import environment
from src.agents import agent
from src.agents.agent_leader import agent_leader
from src.agents.agent_adhoc_q import agent_adhoc
from itertools import permutations


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
        print('Adhoc Prior/Certainty:', adhoc.inference_engine.prior, adhoc.certainty)
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

    env.register_communication(communication_timesteps)

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

    paths = list(set(permutations(l_path))) * repeat

    return paths


# Returns lists of timesteps to complete simulation for each query timestep
def get_query_timesteps(grid_size, stn_pos, tools_pos, l_pos, a_pos, l_tp, all_leader_paths, num_query=None, debug=False):
    max_query = 1
    query = 0

    query_timesteps = []
    while query <= max_query:
        timesteps = []
        for path in all_leader_paths:
            steps = experiment(grid_size, stn_pos, tools_pos, l_pos, a_pos, l_tp, list(path), [query], debug=debug)
            timesteps.append(steps)

        query_timesteps.append(timesteps)

        if query == 0:
            if num_query:
                max_query = num_query - 1
            else:
                max_query = max(timesteps)
        query += 1

    return query_timesteps


def create_graphs(grid_size, stn_pos_perm, stn_names, tools_pos, l_pos, a_pos, l_tp_perm, num_query=None):
    query_timesteps = []
    for stn_pos, l_tp in zip(stn_pos_perm, l_tp_perm):
        all_leader_paths = opt_path_perm(stn_pos, l_pos, l_tp.station_order, 1)
        qt = get_query_timesteps(grid_size, stn_pos, tools_pos, l_pos, a_pos, l_tp, all_leader_paths, num_query)
        query_timesteps.append(qt)
    
    for _ in range(2):
        add = query_timesteps.pop(1)
        for i in range(len(add)):
            query_timesteps[0][i] += add[i]

    num_graphs = 1 # len(stn_pos_perm)
    fig_width = 10
    fig_height = 3 * num_graphs

    fig, ax = plt.subplots(num_graphs, 1, figsize=(fig_width, fig_height))
    fig2, ax2 = plt.subplots(num_graphs, 1, figsize=(fig_width, fig_height))

    for i in range(num_graphs):
        positions = list(range(len(query_timesteps[i])))
        # if num_query:
        #     positions = list(range(num_query))
        labels = positions.copy()
        labels[0] = 'X'

        if num_graphs > 1:
            axs = ax[i]
        else:
            axs = ax
        
        axs.boxplot(query_timesteps[i], positions=positions, whis='range', labels=labels)
        axs.set_title('Timestep Range Based on Query Times: Station %s' % (stn_names[i]))
        axs.set_xlabel('Query Timestep')
        axs.set_ylabel('Timesteps')



        avg = [sum(x) / len(x) for x in query_timesteps[i]]
        err_pos = []
        err_neg = []
        for qt, mu in zip(query_timesteps[i], avg):
            err_p = []
            err_n = []
            for d in qt:
                if d > mu:
                    err_p.append(d - mu)
                else:
                    err_n.append(mu - d)
            
            if len(err_p):
                err_pos.append(sum(err_p) / len(err_p))
            else:
                err_pos.append(0)            
            if len(err_n):
                err_neg.append(sum(err_n) / len(err_n))
            else:
                err_neg.append(0)

        worst_case = [max(qt) for qt in query_timesteps[i]]

        if num_graphs > 1:
            axs2 = ax2[i]
        else:
            axs2 = ax2

        labels = positions.copy()
        labels[0] = 'X'

        axs2.errorbar(positions, worst_case, fmt='ro')
        axs2.errorbar(positions, avg, [err_neg, err_pos], fmt='o', capsize=10)
        axs2.set_title('Timestep Average, Error, and Worst Case: Station %s' % (stn_names[i]))
        axs2.set_xlabel('Query Timestep')
        axs2.set_ylabel('Timesteps')
        axs2.set_xticks(positions)
        axs2.set_xticklabels(labels)
        # axs2.legend(['Worst Case', 'Average Case'])

    fig.tight_layout()
    fig2.tight_layout()

    plt.show()
    # plt.savefig('testgraph')


grid_size = 10
# target station needs to be last listed if you want worst case scenario with wrong inferencing
stn_names = ['1', '2', '3']
stn_pos_perm = [[Point2D(7,2), Point2D(7,8), Point2D(3,8)],
                [Point2D(7,2), Point2D(3,8), Point2D(7,8)],
                [Point2D(3,8), Point2D(7,8), Point2D(7,2)]]
tools_pos = [Point2D(2,3)] # tools_pos needs to be an array but only one tool box is supported so far

l_pos = Point2D(5, 0)
# l_tp = agent.AgentType(len(stn_pos)) # Optional random order of stations to pass to agent_leader()
l_tp_perm = [agent.AgentType([2]), agent.AgentType([2]), agent.AgentType([2])] # Optional fixed order of stations to pass to agent_leader()

a_pos = Point2D(4, 0)

create_graphs(grid_size, stn_pos_perm, stn_names, tools_pos, l_pos, a_pos, l_tp_perm, num_query=24)

path = [gd.Actions.UP] * 3
# experiment(grid_size, stn_pos_perm[2], tools_pos, l_pos, a_pos, l_tp_perm[0], path, [7], debug=True)
