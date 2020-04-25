import numpy as np
from src.utils import astar, first_ind, Point2D
import copy

def gen_graph(obs, width, height):
    epsilon = 1/(width*height+1)
    def graph(node):
        p1,p2 = node
        neighbors = []
        deltas = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
        for d1 in deltas:
            n1 = p1+d1
            if n1[0] < 0 or n1[0] >= width or n1[1] < 0 or n1[1] >= height or n1 in obs:
                continue
            for d2 in deltas:
                if (d1 == 0).all() and (d2 == 0).all():
                    continue
                n2 = p2+d2
                if n2[0] < 0 or n2[0] >= width or n2[1] < 0 or n2[1] >= height or n2 in obs:
                    continue
                if n1 == n2:
                    neighbors.append((2-epsilon, (n1,n2)))
                else:
                    neighbors.append((2, (n1,n2)))
        return neighbors
    return graph

def double_dist(a, b):
    p1, p2 = a
    d1, d2 = b
    return abs(p1[0] - d1[0]) + abs(p1[1] - d1[1]) +\
            abs(p2[0]-d2[0]) + abs(p2[1]-d2[1])
    

def wcd(start, finish1, finish2, obs=[], width=10, height=10):
    return first_ind([a!=b for a,b in astar(gen_graph(obs, width, height),
            (copy.deepcopy(start), copy.deepcopy(start)), (finish1, finish2), h=double_dist)])

