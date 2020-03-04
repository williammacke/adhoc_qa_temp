"""
Plots output of experimental results file
"""
import context
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def graph(args):
    with open(args.data, 'r') as f:
        results = json.load(f)

    plt.rcParams.update({'font.size':18})
    fig, ax = plt.subplots(len(list(results.keys())), 1)

    for i,graph in enumerate(results):
        data = results[graph]
        if args.use_baseline:
            for k in data:
                if k == 'baseline': continue
                for i in range(len(data[k])):
                    data[k][i] -= data['baseline'][i]
        if len(results) > 1:
            axs = ax[i]
        else:
            axs = ax
        avg = {k:sum(data[k])/len(data[k]) for k in data}
        err_pos = {}
        err_neg = {}
        for k in data:
            if k == 'baseline':
                continue
            err_p = []
            err_n = []
            mu = avg[k]
            for p in data[k]:
                if p > mu:
                    err_p.append(p - mu)
                else:
                    err_n.append(mu - p)
            if len(err_p): 
                err_pos[k] = sum(err_p) / len(err_p)
            else:
                err_pos[k] = 0

            if len(err_n):
                err_neg[k] = sum(err_n) / len(err_n)
            else:
                err_neg[k] = 0

        worst_case = {k:min(data[k]) for k in data}
        
        labels = list(k for k in data.keys() if k != 'baseline')

        positions = list(range(len(labels)))

        axs.errorbar(positions, [worst_case[labels[i]] for i in positions], fmt='ro')
        axs.errorbar(positions, [avg[labels[i]] for i in positions],
                [[err_neg[labels[i]] for i in positions], [err_pos[labels[i]] for i in positions]], fmt='o')
        axs.set_xticks(positions)
        axs.set_xticklabels(labels)
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default='results.json', help='datafile to graph')
    parser.add_argument('--use_baseline', action='store_true', help='Flag should be present if baseline is subtracted from results')
    args = parser.parse_args()
    graph(args)




