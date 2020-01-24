from src import environment as env
import numpy as np


myEnv = env.ToolFetchingEnvironment(np.array([0,0]), np.array([1,1]), [np.array([2, 2]), 
    np.array([3,3])], [np.array([5,5]), np.array([6,6])])
myEnv.reset()
myEnv.step([0,0])
