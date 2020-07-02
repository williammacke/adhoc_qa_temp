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

    #Command line arguments
    parser = argparse.ArgumentParser()
    #Experiment input file
    parser.add_argument('-file', '-f', 
                        help="path to experiment file", 
                        required=True)


    args = parser.parse_args()

    #Read file for parameters
    file = open(args.file, "r")
    lines = file.readlines()
    for line in lines:
        #Dimensions for screen
        if line[0] == "d":
            array = list(map(int, line[2:].split(" ")))
            cols = array[0]
            rows = array[1]
        #Fetcher start position
        elif line[0] == "f":
            array = list(map(int, line[2:].split(" ")))   
            fetcher_pos = np.array(array)  
        #Worker start position
        elif line[0] == "w":
            array = list(map(int, line[2:].split(" ")))   
            worker_pos = np.array(array)
        #Number of stations
        elif line[0] == "n":
            num = 3
        #List of Station Positions
        elif line[0] == "s":
            array = list(map(int, line[2:].split(" ")))
            stn_pos = []
            for i in range(num):
                stn_pos.append(np.array([array[i * 2], array[i * 2 + 1]]))
        #List of tool positions
        elif line[0] == "t":
            array = list(map(int, line[2:].split(" ")))
            tool_pos = []
            for i in range(num):
                tool_pos.append(np.array([array[i * 2], array[i * 2 + 1]]))
        #Goal station
        elif line[0] == "g":
            array = list(map(int, line[2:].split(" ")))
            goal_stn = array[0]

    env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, goal_stn)
    #grab observation
    obs = env.reset()
    done = [False, False]
    fetcher = FetcherQueryPolicy()
    worker = RandomWorkerPolicy()

    gui = GUI(cols, rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos)

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
