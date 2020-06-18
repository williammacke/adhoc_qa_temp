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

# class Input(enum.Enum):
#   W = 2
#   A = 1
#   S = 3
#   D = 0
#   J = 5

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
    # fetcher_pos = np.array([4, 0])
    fetcher_pos = np.array([0, 3])
    #Worker Start Position
    # worker_pos = np.array([5, 0])
    worker_pos = np.array([0, 2])
    #List of Station Positions
    # stn_pos = [np.array([7,2]), np.array([7,8]), np.array([3, 8])]
    stn_pos = [np.array([2,0]), np.array([9,0]), np.array([9,4])]
    #List of tool positions, in this example they are all located in the same spot
    # tool_pos = [np.array([2,3]) for _ in range(3)]
    tool_pos = [np.array([3,5]) for _ in range(3)]
    goal_stn = 0
    env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, goal_stn)
    #grab observation
    obs = env.reset()
    done = [False, False]
    fetcher = FetcherQueryPolicy()
    worker = RandomWorkerPolicy()
    #Print goal/positions for human to determine where to go
    # print("The fetcher is at", fetcher_pos)
    # print("You (the worker) is at", worker_pos)
    # print("The fetcher's tool box is at", tool_pos[0])
    print("Your goal station is station", goal_stn)
    print("Press\n W - up\n A - left\n S - down\n D - right\n J - done (press when arrived at station)\n Z - exit")

    gui = GUI(10, 6, stn_pos, tool_pos, worker_pos, fetcher_pos)

    #run until done
    while not done[0]:
        # only needed for rendering
        env.render()
        

        sleep(0.05)

        fetcher_move = fetcher(obs[1])
        returnVal = gui.on_execute(fetcher_move[0]) # returns with input value
        if(returnVal == Input.Exit):
            break
        
        obs, reward, done, _ = env.step([returnVal.value, fetcher_move])
        # obs, reward, done, _ = env.step([get_worker_action(), fetcher(obs[1])])

    env.close()
    gui.on_cleanup()
