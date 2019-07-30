import unittest
from src import global_defs
from src.agents import agent_random
import pdb
from src import utils

import numpy as np



class agent_random_test(unittest.TestCase):
    def test_agent_respond_sanity(self):
        """
        Testing that agent probabilites always sum to one and that the agent always takes actions within bounds.
        :return:
        """

        for i in range(1000):
            with self.subTest(i=i):
                random_pos_array = np.random.randint(0,10,(20,2)) #Generating 20 random locations
                random_pos_list = [global_defs.Point2D(ele[0],ele[1]) for ele in random_pos_array]
                a = agent_random.agent_random(global_defs.Point2D(random_pos_list[0][0],random_pos_list[0][1]))


                allPos = random_pos_list
                myInd = 0
                loadIndices = range(4,8)
                random_observation = global_defs.obs(allPos,myInd,loadIndices)

                (action_probs,action_idx) = a.respond(random_observation)

                self.assertTrue(len(action_probs)==6)
                np.testing.assert_approx_equal(np.sum(action_probs),1)
                self.assertTrue(action_idx<6)
                print(action_probs,action_idx)
                if action_idx==global_defs.Actions.LOAD:
                    is_neighbor=False
                    for loadidx in (loadIndices):
                        loadPos = allPos[loadidx]
                        is_neighbor = is_neighbor or utils.is_neighbor(loadPos,a.pos)
                    self.assertTrue(is_neighbor)

                is_neighbor = False
                for loadidx in (loadIndices):
                    loadPos = allPos[loadidx]
                    is_neighbor = is_neighbor or utils.is_neighbor(loadPos, a.pos)
                if is_neighbor:
                    msg_str= "Is neighbor {} {}".format(a.pos,[allPos[i] for i in range(4,8)])
                    self.assertTrue(action_probs[-1]>0,msg=msg_str)


