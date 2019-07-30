import unittest
import pdb

from src import environment
from src.agents import agent_lifter
import random
import numpy as np
from src import global_defs
from src import utils

class env_test(unittest.TestCase):
    def test_sameType(self):
        n_tests = 0
        total_test = 100
        for i in range(total_test):
            #Agents: Same type.
            #Object: Not on edges.
            #Distance: target_dist steps.

            object_location = np.array(random.sample(range(1,8),2)[:2])

            #Put agents target_dist positions away.
            target_dist = 6
            displacement = np.random.randint(-target_dist,target_dist,(2,2))
            displacement[:,1] = random.choice([-1,1])*(target_dist-np.abs(displacement[:,0]))

            a1_pos = (object_location[0]+displacement[0][0],object_location[1]+displacement[0][1])
            a2_pos = (object_location[0]+displacement[1][0],object_location[1]+displacement[1][1])

            if not(utils.check_within_boundaries(a1_pos) and utils.check_within_boundaries(a2_pos)):
                continue
            else:
                n_tests+=1
                print("-----------Test Iter: {}-------------".format(i))
                with self.subTest(i=i,msg='Experiment {}'.format(i)):
                    self.assertTrue(np.sum(np.abs(object_location-a1_pos))==target_dist)
                    self.assertTrue(np.sum(np.abs(object_location-a2_pos))==target_dist)

                    a1 = agent_lifter.agent_lifter(a1_pos,1)
                    a2 = agent_lifter.agent_lifter(a2_pos,1)

                    object_location = global_defs.Point2D(object_location[0],object_location[1])

                    env = environment.environment(10,[object_location,global_defs.Point2D(0,0)],False)
                    env.register_agent(a1)
                    env.register_agent(a2)

                    n_steps = 0
                    #print("Target Distance: {}".format(target_dist))
                    #print("Object location {}".format(object_location))
                    print("A1 {}".format(a1.pos))
                    print("A2 {}".format(a2.pos))
                    while(not env.is_terminal and n_steps<10):
                        is_terminal, reward = env.step()
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

    def test_diffType(self):
        n_tests = 0
        total_test = 100
        for i in range(total_test):
            #Agents: Same type.
            #Object: Not on edges.
            #Distance: target_dist steps.

            object_location = np.array(random.sample(range(1,8),4)[:2])

            #Put agents target_dist positions away.
            target_dist = 6
            displacement = np.random.randint(-target_dist,target_dist,(2,2))
            displacement[:,1] = random.choice([-1,1])*(target_dist-np.abs(displacement[:,0]))

            a1_pos = (object_location[0]+displacement[0][0],object_location[1]+displacement[0][1])
            a2_pos = (object_location[0]+displacement[1][0],object_location[1]+displacement[1][1])

            if not(utils.check_within_boundaries(a1_pos) and utils.check_within_boundaries(a2_pos)):
                continue
            else:
                n_tests+=1
                print("-----------Test Iter: {}-------------".format(i))
                with self.subTest(i=i,msg='Experiment {}'.format(i)):
                    self.assertTrue(np.sum(np.abs(object_location-a1_pos))==target_dist)
                    self.assertTrue(np.sum(np.abs(object_location-a2_pos))==target_dist)

                    a1 = agent_lifter.agent_lifter(a1_pos,1)
                    a2 = agent_lifter.agent_lifter(a2_pos,2)

                    object_location = global_defs.Point2D(object_location[0],object_location[1])

                    env = environment.environment(10,[object_location,global_defs.Point2D(0,0)],False)
                    env.register_agent(a1)
                    env.register_agent(a2)

                    n_steps = 0
                    print("******** Test 2 **************8")
                    print("Target Distance: {}".format(target_dist))
                    print("Object location {}".format(object_location))

                    while(not env.is_terminal and n_steps<100):
                        is_terminal,reward = env.step()
                        n_steps+=1

                    if is_terminal:
                        print(n_steps)
                        print("A1 {}".format(a1))
                        print("A2 {}".format(a2))
                        print("Env {}".format(env))
                        print("Reward {} ".format(reward))
                    print("******************************")
                #self.assertTrue(reward==0)
        print("n_tests {} total_tests {}".format(n_tests,total_test))

class _env_copy_test(unittest.TestCase):
    def _test_sameType_copy_everyStep(self):
        n_tests = 0
        total_test = 100
        for i in range(total_test):
            #Agents: Same type.
            #Object: Not on edges.
            #Distance: target_dist steps.

            object_location = np.array(random.sample(range(1,8),2)[:2])

            #Put agents target_dist positions away.
            target_dist = 6
            displacement = np.random.randint(-target_dist,target_dist,(2,2))
            displacement[:,1] = random.choice([-1,1])*(target_dist-np.abs(displacement[:,0]))

            a1_pos = (object_location[0]+displacement[0][0],object_location[1]+displacement[0][1])
            a2_pos = (object_location[0]+displacement[1][0],object_location[1]+displacement[1][1])

            if not(utils.check_within_boundaries(a1_pos) and utils.check_within_boundaries(a2_pos)):
                continue
            else:
                n_tests+=1
                print("-----------Test Iter: {}-------------".format(i))
                with self.subTest(i=i,msg='Experiment {}'.format(i)):
                    self.assertTrue(np.sum(np.abs(object_location-a1_pos))==target_dist)
                    self.assertTrue(np.sum(np.abs(object_location-a2_pos))==target_dist)

                    a1 = agent_lifter.agent_lifter(a1_pos,1)
                    a2 = agent_lifter.agent_lifter(a2_pos,1)

                    object_location = global_defs.Point2D(object_location[0],object_location[1])

                    env = environment.environment(10,[object_location,global_defs.Point2D(0,0)],False)
                    env.register_agent(a1)
                    env.register_agent(a2)

                    n_steps = 0
                    #print("Target Distance: {}".format(target_dist))
                    #print("Object location {}".format(object_location))
                    print("A1 {}".format(a1.pos))
                    print("A2 {}".format(a2.pos))
                    while(not env.is_terminal and n_steps<10):
                        new_env = env.__copy__()

                        is_terminal, reward = env.step()
                        is_terminal2, reward2 = new_env.step()

                        self.assertEqual(is_terminal,is_terminal2)
                        self.assertEqual(reward,reward2)

                        s1 = env.__getstate__()
                        s2 = new_env.__getstate__()
                        try:
                            self.assertTrue(utils.compare_env_states(s1,s2))
                        except:
                            pdb.set_trace()

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

    def _test_sameType_copy_beginning(self):
        n_tests = 0
        total_test = 100
        for i in range(total_test):
            # Agents: Same type.
            # Object: Not on edges.
            # Distance: target_dist steps.

            object_location = np.array(random.sample(range(1, 8), 2)[:2])

            # Put agents target_dist positions away.
            target_dist = 6
            displacement = np.random.randint(-target_dist, target_dist, (2, 2))
            displacement[:, 1] = random.choice([-1, 1]) * (target_dist - np.abs(displacement[:, 0]))

            a1_pos = (object_location[0] + displacement[0][0], object_location[1] + displacement[0][1])
            a2_pos = (object_location[0] + displacement[1][0], object_location[1] + displacement[1][1])

            if not (utils.check_within_boundaries(a1_pos) and utils.check_within_boundaries(a2_pos)):
                continue
            else:
                n_tests += 1
                print("-----------Test Iter: {}-------------".format(i))
                with self.subTest(i=i, msg='Experiment {}'.format(i)):
                    self.assertTrue(np.sum(np.abs(object_location - a1_pos)) == target_dist)
                    self.assertTrue(np.sum(np.abs(object_location - a2_pos)) == target_dist)

                    a1 = agent_lifter.agent_lifter(a1_pos, 1)
                    a2 = agent_lifter.agent_lifter(a2_pos, 1)

                    object_location = global_defs.Point2D(object_location[0], object_location[1])

                    env = environment.environment(10, [object_location, global_defs.Point2D(0, 0)], False)
                    env.register_agent(a1)
                    env.register_agent(a2)

                    n_steps = 0
                    # print("Target Distance: {}".format(target_dist))
                    # print("Object location {}".format(object_location))
                    print("A1 {}".format(a1.pos))
                    print("A2 {}".format(a2.pos))

                    new_env = env.__copy__()

                    while (not env.is_terminal and n_steps < 10):

                        is_terminal, reward = env.step()
                        is_terminal2, reward2 = new_env.step()

                        self.assertEqual(is_terminal,is_terminal2)
                        self.assertEqual(reward,reward2)

                        s1 = env.__getstate__()
                        s2 = new_env.__getstate__()
                        self.assertTrue(utils.compare_env_states(s1, s2))

                        n_steps += 1

                    if is_terminal:
                        print("Terminal Reward {}".format(reward))
                        print("N_iters {}".format(n_steps))
        # print("n_tests {} total_tests {}".format(n_tests,total_test))
    def _test_sameType_copy_middleStep(self):
        n_tests = 0
        total_test = 100
        for i in range(total_test):
            #Agents: Same type.
            #Object: Not on edges.
            #Distance: target_dist steps.

            object_location = np.array(random.sample(range(1,8),2)[:2])

            #Put agents target_dist positions away.
            target_dist = 6
            displacement = np.random.randint(-target_dist,target_dist,(2,2))
            displacement[:,1] = random.choice([-1,1])*(target_dist-np.abs(displacement[:,0]))

            a1_pos = (object_location[0]+displacement[0][0],object_location[1]+displacement[0][1])
            a2_pos = (object_location[0]+displacement[1][0],object_location[1]+displacement[1][1])

            if not(utils.check_within_boundaries(a1_pos) and utils.check_within_boundaries(a2_pos)):
                continue
            else:
                n_tests+=1
                print("-----------Test Iter: {}-------------".format(i))
                with self.subTest(i=i,msg='Experiment {}'.format(i)):
                    self.assertTrue(np.sum(np.abs(object_location-a1_pos))==target_dist)
                    self.assertTrue(np.sum(np.abs(object_location-a2_pos))==target_dist)

                    a1 = agent_lifter.agent_lifter(a1_pos,1)
                    a2 = agent_lifter.agent_lifter(a2_pos,1)

                    object_location = global_defs.Point2D(object_location[0],object_location[1])

                    env = environment.environment(10,[object_location,global_defs.Point2D(0,0)],False)
                    env.register_agent(a1)
                    env.register_agent(a2)

                    n_steps = 0
                    #print("Target Distance: {}".format(target_dist))
                    #print("Object location {}".format(object_location))
                    print("A1 {}".format(a1.pos))
                    print("A2 {}".format(a2.pos))
                    _,_ = env.step()
                    _,_ = env.step()

                    new_env = env.__copy__()
                    while(not env.is_terminal and n_steps<10):

                        is_terminal, reward = env.step()
                        is_terminal2, reward2 = new_env.step()

                        self.assertEqual(is_terminal,is_terminal2)
                        self.assertEqual(reward,reward2)

                        s1 = env.__getstate__()
                        s2 = new_env.__getstate__()
                        try:
                            self.assertTrue(utils.compare_env_states(s1,s2))
                        except:
                            pdb.set_trace()

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
