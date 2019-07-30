import unittest
from src import global_defs
from src import utils
import ipdb
from src.astar.pyastar import astar

import numpy as np


class astar_test(unittest.TestCase):
    def test_without_obstacles(self):
        """
        Testing that ASTAR finds path when it exists for sure by having no obstacles at all.
        The start and stop position are very far away and we are not testing edge conditions.
        :return:
        """

        time_array = []
        for i in range(100):
            with self.subTest(i=i):
                start_position = np.array([8, np.random.randint(0, 7)])
                goal_position = np.array([np.random.randint(0, 7), 8])
                obstacles = []

                start_position = tuple(start_position)
                goal_position = tuple(goal_position)

                #begin_time = time.time()
                #ipdb.set_trace()
                a = astar(start_position, goal_position, obstacles, 10)
                #time_array.append(time.time() - begin_time)
                res = a.find_minimumpath()

                #for node in res[1]:
                #    print(node)
                #print('------')

                #pdb.set_trace()
                self.assertEqual(res[0],1)

    def test_with_obstacles(self):
        """
        Testing that ASTAR should work even when there are obstacles.
        The start and end are very far away, so we are not testing edge cases.
        :return:
        """
        time_array = []
        for i in range(100):
            with self.subTest(i=i):
                start_position = np.array([8, np.random.randint(0, 7)])
                goal_position = np.array([np.random.randint(0, 7), 8])

                start_position = tuple(start_position)
                goal_position = tuple(goal_position)


                obstacle_array = np.random.randint(0,10,(10,2))
                obstacles =[]
                for obstacle in obstacle_array:
                    if np.all(obstacle==start_position) or np.all(obstacle==goal_position):
                        pass
                    else:
                        obstacles.append((obstacle[0],obstacle[1]))


                # begin_time = time.time()
                #ipdb.set_trace()
                a = astar(start_position, goal_position, obstacles, 10)
                # time_array.append(time.time() - begin_time)
                res = a.find_minimumpath()
                obstacle_set = [global_defs.Point2D(obstacle[0],obstacle[1]) for obstacle in obstacles]

                self.assertEqual(res[0], 1)

                #print(obstacle_array)
                for node in res[1][:-1]:
                    if (global_defs.Point2D(node[0],node[1]) in obstacle_set):
                        print('----'+str(global_defs.Point2D(node[0],node[1])))

                    self.assertFalse(global_defs.Point2D(node[0],node[1]) in obstacle_set)

    def test_with_obstacles_noroute(self):
        """
        Testing with no route to the end.
        :return:
        """
        time_array = []
        for i in range(100):
            with self.subTest(i=i):
                start_position = np.array([1, np.random.randint(0, 7)])
                goal_position = np.array([7,np.random.randint(0, 7)])

                obstacle_array = np.array([[4,i] for i in range(0,10)])
                start_position = tuple(start_position)
                goal_position = tuple(goal_position)

                obstacles = []
                for obstacle in obstacle_array:
                    if np.all(obstacle == start_position) or np.all(obstacle == goal_position):
                        pass
                    else:
                        obstacles.append((obstacle[0], obstacle[1]))

                # begin_time = time.time()
                a = astar(start_position, goal_position, obstacles, 10)
                # time_array.append(time.time() - begin_time)
                res = a.find_minimumpath()

                obstacle_set = [global_defs.Point2D(obstacle[0],obstacle[1]) for obstacle in obstacles]

                self.assertEqual(res[0], 0) #Path not found.


    def test_immediate_neighbor(self):
        """
        Testing ASTAR when the start and end are immediate neighbors of each other. Edge Case.
        :return:
        """
        for i in range(100):
            with self.subTest(i=i):
                start_position = np.array([np.random.randint(1,7), np.random.randint(1, 7)])
                p = np.random.randint(0,1)
                goal_position = np.power(-1,np.random.randint(1,2))*np.array([p,1-p])+start_position
                #print(goal_position,start_position)
                obstacles = []
                start_position = (start_position[0],start_position[1])
                goal_position = (goal_position[0],goal_position[1])
                # begin_time = time.time()
                a = astar(start_position, goal_position, obstacles, 10, False)
                # time_array.append(time.time() - begin_time)
                res = a.find_minimumpath()

                # pdb.set_trace()
                start_position = global_defs.Point2D(start_position[0],start_position[1])
                goal_position = global_defs.Point2D(goal_position[0],goal_position[1])
                self.assertTrue(utils.is_neighbor(start_position,goal_position))
                self.assertEqual(res[0], 1)
                self.assertTrue(len(res[1])==2)


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
