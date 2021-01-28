import context
import unittest
import src.agents.query_policies as qp
import numpy as np


class UtilsTest(unittest.TestCase):
    def test_is_ZB(self):
        w_pos = np.array([0, 0])
        f_pos = np.array([10, 10])
        s_pos = [np.array([5,5]), np.array([7, 7])]
        t_pos = [np.array([5, 5]), np.array([7, 7])]
        f_tool = None
        w_action = None
        f_action = None
        answer = None
        answer1 = qp.is_ZB((w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer), 0, 1)
        self.assertTrue(not answer1)
        f_pos = np.array([6,6])
        answer2 = qp.is_ZB((w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer), 0, 1)
        self.assertTrue(answer2)
        f_pos = np.array([5,7])
        answer3 = qp.is_ZB((w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer), 0, 1)
        self.assertTrue(answer3)

    #def 




if __name__ == '__main__':
    unittest.main()
