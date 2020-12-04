"""
This is the working file for the Human Fetcher Collaboration
"""
#gives file access to other src directory
import context
import numpy as np
import enum

from time import sleep

from src.environment import ToolFetchingEnvironment
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent_adhoc_q import FetcherQueryPolicy
from HRI.pygame_gui import GUI, Input

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return bytes.decode(msvcrt.getch())

arrived = False

def get_worker_action():
    global arrived
    #fill in code for human input here
    while (True):
        if (arrived):
            return Input["J"].value
        getch = _Getch()
        val = getch().upper()
  
        if (val in Input.__members__):
            if(val == "J"):
                arrived = True
            return Input[val].value
        elif (val == "Z"):
            exit()
        else: 
            print ("Not a valid input, please try again.")

if __name__ == '__main__':
    #Fetcher start position
    fetcher_pos = np.array([0, 3])
    #Worker Start Position
    worker_pos = np.array([0, 2])
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