"""
This is the file for running different human experiments
"""
#gives file access to other src directory
import context
import numpy as np
import enum
import argparse

from time import sleep

from src.environment import ToolFetchingEnvironment
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent_adhoc_q import FetcherQueryPolicy
from HRI.pygame_gui import GUI, Input

arrived = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fetcher', '-f', 
                        type=int,
                        nargs=2,
                        help="type a comma separated value for the fetcher location", 
                        required=True)
    parser.add_argument('-worker', '-w',
                        type=int,
                        nargs=2, 
                        help="type a comma separated value for the worker location", 
                        required=True)

    args = parser.parse_args()
    
    #Fetcher start position
    fetcher_pos = np.array(args.fetcher)
    #Worker Start Position
    worker_pos = np.array(args.worker)
    #List of Station Positions
    stn_pos = [np.array([2,0]), np.array([9,0]), np.array([9,4])]
    #List of tool positions, in this example they are all located in the same spot
    tool_pos = [np.array([3,5]) for _ in range(3)]
    goal_stn = 1
    env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, goal_stn)
    #grab observation
    obs = env.reset()
    done = [False, False]
    fetcher = FetcherQueryPolicy()
    worker = RandomWorkerPolicy()

    gui = GUI(10, 6, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos)

    #run until done
    while not done[0]:
        # only needed for rendering

        fetcher_move = fetcher(obs[1])
        returnVal = gui.on_execute(fetcher_move[0]) # returns with input value
        # Exit experiment
        if(returnVal == Input.Exit):
            break

        sleep(0.05)
        obs, reward, done, _ = env.step([returnVal.value, fetcher_move])
        # obs, reward, done, _ = env.step([get_worker_action(), fetcher(obs[1])])

    gui.on_cleanup()
