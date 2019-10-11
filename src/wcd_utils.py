from collections import deque
from heapq import heappush, heappop


def adj(p):
    x,y = p
    return [(x+1,y), (x-1, y), (x,y+1), (x,y-1), (x,y)]

def genGraph(state):
    epsilon = 1/(len(state)+len(state[0])+1)
    def graph(node):
        p1,p2 = node
        l = []
        for pp1 in adj(p1):
            x1,y1 = pp1
            if x1 < 0  or x1 >= len(state) or y1 < 0 or y1 >= len(state[0]):
                continue
            for pp2 in adj(p2):
                if pp1 == p1 and pp2 == p2:
                    continue
                x2,y2 = pp2
                if x2 < 0  or x2 >= len(state) or y2 < 0 or y2 >= len(state[0]):
                    continue
                if pp1 == pp2:
                    if not state[x1][y1]:
                        l.append((2-epsilon,(pp1,pp2)))
                else:
                    if not state[x1][y1] and not state[x2][y2]:
                        l.append((2,(pp1,pp2)))
        return l
    return graph




def wcd_astar(graph, start, finish, h):
    q = [(0, 0, start, [start])]
    visited = set()
    while q:
        e,val,state,plan = heappop(q)
        if state == finish:
            return plan
        if state in visited: continue
        visited.add(state)
        for cost, sp in graph(state):
            heappush(q, (val+cost+h(sp,finish), val+cost, sp, plan+[sp]))

def pruned_reduce(state, loc):
    wcd = 0
    plan = []
    closed = set()
    Q = deque()
    Q.append(plan)
    while Q:
        A = Q.popleft()

