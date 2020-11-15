import context
from src.environment import ToolFetchingEnvironment
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import os



def graph(args):
    results = {}
    for n in os.listdir(args.data):
        with open(args.data+'/'+n, 'r') as f:
            results[n] = json.load(f)

    c = args.cost
    dist = args.distribution


    data = []

    key = None
    num_bins = 0

    for v in results:
        if dist not in v:
            continue
        if c not in v:
            continue
        key = v
        labels = list(results[key]['fetcher_actions'].keys())
        for k in labels:
            l = []
            print(k, len(results[v]['fetcher_actions'][k]))
            for actions in results[v]['fetcher_actions'][k]:
                a = np.array(actions)
                num_bins = max(num_bins, len(a))
                print(a)
                l += list(np.where(a == ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY)[0])
            data.append(l)
    print(data)
    d = np.array(data[1])
    print(np.where(d == 0))
    print(int(ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY))

    plt.hist(data, label=labels, bins=num_bins+1)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help = 'data directory')
    parser.add_argument('-c', '--cost', help='Cost to use')
    parser.add_argument('-d', '--distribution', help='Distribution to use')
    args = parser.parse_args()
    graph(args)

            

