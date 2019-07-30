import unittest
from src import global_defs
from src import utils
from src.agents import agent_lifter
import pdb
from src.astar2 import astar

import numpy as np


class utiltest(unittest.TestCase):
    def test_neighborhood_check(self):
        """
        Test for is_neighbor function in utils.py
        :return:
        """
        #positive examples
        points = []
        for x in range(0,10):
            for y in range(0,10):
                points.append(global_defs.Point2D(x,y))

        neighbors = []
        for point in points:
            curr_neighbors = []
            for move in global_defs.MOVES:
                curr_neighbors.append(point+move)
            neighbors.append(curr_neighbors)

        not_neighbors = []
        for point in points:
            curr_not_neighbors = []
            for move in global_defs.MOVES:
                curr_not_neighbors.append(point+2*(move)+2)
            not_neighbors.append(curr_not_neighbors)

        for neighbor_set,point in zip(neighbors,points):
            for neighbor in neighbor_set:
                self.assertTrue(utils.is_neighbor(neighbor,point))
                self.assertTrue(utils.is_neighbor(point,neighbor))

        for neighbor_set,point in zip(not_neighbors,points):
            for neighbor in neighbor_set:
                #pdb.set_trace()
                self.assertFalse(utils.is_neighbor(neighbor,point))
                self.assertFalse(utils.is_neighbor(point,neighbor))


class agent_lifter_test(unittest.TestCase):
    def test_agent_tp_1_respond(self):
        random_pos_array = np.random.randint(0,10,(20,2)) #Generating 20 random locations
        random_pos_list = [global_defs.Point2D(ele[0],ele[1]) for ele in random_pos_array]
        a = agent_lifter.agent_lifter(global_defs.Point2D(random_pos_list[0][0],random_pos_list[0][1]),1)

        allPos = random_pos_list
        myInd = 0
        loadIndices = range(4,8)
        random_observation = global_defs.obs(allPos,myInd,loadIndices)

        (action_probs,action_idx) = a.respond(random_observation)

        self.assertTrue(len(action_probs)==6)
        np.testing.assert_approx_equal(np.sum(action_probs),1)
        self.assertTrue(action_idx<6)

class astar_test(unittest.TestCase):
    def test_without_obstacles(self):
        time_array = []
        for i in range(100):
            with self.subTest(i=i):
                start_position = np.array([8, np.random.randint(0, 7)])
                goal_position = np.array([np.random.randint(0, 7), 8])
                obstacles = []

                #begin_time = time.time()
                a = astar(start_position, goal_position, obstacles, 10, False)
                #time_array.append(time.time() - begin_time)
                res = a.find_minimumpath()

                #for node in res[1]:
                #    print(node)
                #print('------')

                #pdb.set_trace()
                self.assertEqual(res[0],1)

    def test_with_obstacles(self):
        time_array = []
        for i in range(100):
            with self.subTest(i=i):
                start_position = np.array([8, np.random.randint(0, 7)])
                goal_position = np.array([np.random.randint(0, 7), 8])

                obstacle_array = np.random.randint(0,10,(10,2))
                obstacles =[]
                for obstacle in obstacles:
                    if np.all(obstacle==start_position) or np.all(obstacle==goal_position):
                        pass
                    else:
                        obstacles.append(global_defs.Point2D(obstacle[0],obstacle[1]))


                # begin_time = time.time()
                a = astar(start_position, goal_position, obstacles, 10, False)
                # time_array.append(time.time() - begin_time)
                res = a.find_minimumpath()
                a.print_path(res)
                for node in res[1][:-1]:
                    self.assertFalse(node.is_obstacle)
                #self.assertEqual(res[0], 1)

    def test_with_obstacles_noroute(self):
        time_array = []
        for i in range(100):
            with self.subTest(i=i):
                start_position = np.array([1, np.random.randint(0, 7)])
                goal_position = np.array([7,np.random.randint(0, 7)])

                obstacle_array = np.array([[4,i] for i in range(0,11)])

                obstacles = []
                for obstacle in obstacle_array:
                    if np.all(obstacle == start_position) or np.all(obstacle == goal_position):
                        pass
                    else:
                        obstacles.append(global_defs.Point2D(obstacle[0], obstacle[1]))

                # begin_time = time.time()
                a = astar(start_position, goal_position, obstacles, 10, False)
                # time_array.append(time.time() - begin_time)
                res = a.find_minimumpath()
                a.print_path(res)
                for node in res[1][:-1]:
                    self.assertFalse(node.is_obstacle)


                self.assertEqual(res[0],0)
                #self.assertEqual(res[0], 1)

    def test_immediate_neighbor(self):
        for i in range(100):
            with self.subTest(i=i):
                start_position = np.array([np.random.randint(1,7), np.random.randint(1, 7)])
                p = np.random.randint(0,1)
                goal_position = np.power(-1,np.random.randint(1,2))*np.array([p,1-p])+start_position
                #print(goal_position,start_position)
                obstacles = []
                start_position = global_defs.Point2D(start_position[0],start_position[1])
                goal_position = global_defs.Point2D(goal_position[0],goal_position[1])
                # begin_time = time.time()
                a = astar(start_position, goal_position, obstacles, 10, False)
                # time_array.append(time.time() - begin_time)
                res = a.find_minimumpath()
                a.print_path(res)

                # pdb.set_trace()
                self.assertTrue(utils.is_neighbor(start_position,goal_position))
                self.assertEqual(res[0], 1)


class environment_test(unittest.TestCase):
    #Methods to properly test environment.
    #But how do we test the environment to begin with?
    def test_termination_check(self):
        for i in range(100):
            with self.subTest(i=i):
                self.assertTrue(True)



if __name__ == '__main__':
    # Run only the tests in the specified classes
    test_classes_to_run = [astar_test]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
