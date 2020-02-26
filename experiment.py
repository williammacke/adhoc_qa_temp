from src import environment as env
import numpy as np
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent_adhoc_q import FetcherQueryPolicy
from src.agents.agent_adhoc_q import FetcherAltPolicy


myEnv = env.ToolFetchingEnvironment(np.array([0,0]), np.array([1,1]), [np.array([2, 2]), 
    np.array([3,3])], [np.array([5,5]), np.array([2,5])], 0)
obs = myEnv.reset()
fetcher = FetcherAltPolicy()
worker = RandomWorkerPolicy()
done = [False, False]
while not done[0]:
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, goal = obs[0]
    print(f"worker position {w_pos}")
    print(f"worker action {w_action}")
    print(f"goal position {s_pos[goal]}")
    print(f"goal {goal}")
    print(f"fetcher position {f_pos}")
    print(f"fetcher action {f_action}")
    print(f"fetcher tool {f_tool}")
    myEnv.render()
    input()
    obs, reward, done, data = myEnv.step([worker(obs[0]),fetcher(obs[1])])

myEnv.close()
