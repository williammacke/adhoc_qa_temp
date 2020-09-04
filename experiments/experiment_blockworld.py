import context
from src.environment2 import BlockWorld4Teams
from src.agents.BW4T_agent import GreedyAgent
import numpy as np



def exp(args):
    num_rooms = 6
    drop_room = 0
    num_blocks = 5
    locations = np.array([1, 2, 3, 4, 5])
    order = np.array([0,1,2,3,4])
    num_agents = 2
    agents = [GreedyAgent(), GreedyAgent()]
    env = BlockWorld4Teams(num_rooms, drop_room, num_blocks, locations, order, num_agents, agents)
    state = env.reset()
    done = False
    while not done:
        print(state[0])
        print(state[1])
        print(state[2])
        print(done)
        input()
        state, reward, done, _ = env.step()
    print(state[0])
    print(state[1])
    print(state[2])
    print(done)
    input()


if __name__ == '__main__':
    exp({})
