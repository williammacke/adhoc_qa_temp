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



def get_worker_action():
    #fill in code for human input here
    while (True):
      print("Press W - up, A - left, S - down, D - right: ")
      getch = _Getch()
      val = getch().upper()
      # val = input("Press W - up, A - left, S - down, D - right: ")
      print(val)
      if (val == "W"):
        return 2
      elif (val == "A"):
        return 1
      elif (val == "S"):
        return 3
      elif (val == "D"):
        return 0
      else: 
        quit()
        print ("Could not determine your input, try again.")




if __name__ == '__main__':
    #Fetcher start position
    fetcher_pos = np.array([4, 0])
    #Worker Start Position
    worker_pos = np.array([5, 0])
    #List of Station Positions
    stn_pos = [np.array([7,2]), np.array([7,8]), np.array([3, 8])]
    #List of tool positions, in this example they are all located in the same spot
    tool_pos = [np.array([2,3]) for _ in range(3)]
    env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, 2)
    #grab observation
    obs = env.reset()
    done = [False, False]
    fetcher = FetcherQueryPolicy()
    worker = RandomWorkerPolicy()
    #Print goal/positions for human to determine where to go
    print("The worker may go to the stations at", stn_pos[0], stn_pos[1], stn_pos[2])
    print("Your tool box is at", tool_pos[0])

    #run until done
    while not done[0]:
        #only needed for rendering
        print("The fetcher is at", fetcher_pos)
        print("You (the worker) is at", worker_pos)
        print(fetcher(obs[1]))
        env.render()
        sleep(0.05)
        obs, reward, done, _ = env.step([get_worker_action(), fetcher(obs[1])])
    env.close()