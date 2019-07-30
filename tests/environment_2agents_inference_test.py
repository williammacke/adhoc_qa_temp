import unittest
import pdb
import time
import ipdb
from src import environment
from src.agents import agent_lifter
from src.agents import agent_adhoc
import random
import numpy as np
from src import global_defs
from src import utils

class env_test(unittest.TestCase):
    def test_sameType(self):
        n_tests = 0
        total_test = 1
        for i in range(total_test):
            #Agents: Same type.
            #Object: Not on edges.
            #Distance: target_dist steps.

            obj_pos,agent_pos = utils.generate_initial_conditions(global_defs.N_TYPES,2)
            #ipdb.set_trace()
            if True:
                n_tests+=1
                print("-----------Test Iter: {}-------------".format(i))
                with self.subTest(i=i,msg='Experiment {}'.format(i)):

                    a1 = agent_lifter.agent_lifter(agent_pos[0],2)
                    a2 = agent_lifter.agent_lifter(agent_pos[1],2)
                    a3 = agent_adhoc.agent_adhoc(a2.pos)

                    env = environment.environment(global_defs.GRID_SIZE,obj_pos,False)

                    env.register_agent(a1)
                    env.register_agent(a2)

                    n_steps = 0
                    #print("Target Distance: {}".format(target_dist))
                    #print("Object location {}".format(object_location))
                    print("A1 {}".format(a1.pos))
                    print("A2 {}".format(a2.pos))
                    while(not env.is_terminal ):
                        is_terminal, reward = env.step()
                        #ipdb.set_trace()
                        time.sleep(0.1)
                        tp_estimate = a3.respond(env)
                        print("TPESTIMATE STEP",tp_estimate,env.step_count)
                        if env.step_count==5:
                            a1.tp=1
                        n_steps+=1
                        #print(n_steps)
                        #print("A1 {}".format(a1.pos))
                        #print("A2 {}".format(a2.pos))
                        #if is_terminal:
                            #print("Reward {}".format(reward))
                    if is_terminal:
                        print("Terminal Reward {}".format(reward))
                        print("N_iters {}".format(n_steps))
        #print("n_tests {} total_tests {}".format(n_tests,total_test))



if __name__=='__main__':
    test_classes_to_run = [env_test]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
