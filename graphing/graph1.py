import json
import matplotlib.pyplot as plt
import numpy as np


def graph(args):
    with open('results.json', 'r') as f:
        results = json.load(f)

    plt.rcParams.update({'font.size':18})
    fig, ax = plt.subplots(len(list(results.keys())), 1)

    for i,graph in enumerate(results):
        data = results[graph]
        axs = ax[i]
        avg = {k:sum(data[k])/len(data[k]) for k in data}
        err_pos = {}
        err_neg = {}
        for k in data:
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
        
        labels = list(data.keys())

        positions = list(range(len(data)))

        axs.errorbar(positions, [worst_case[labels[i]] for i in positions], fmt='ro')
        axs.errorbar(positions, [avg[labels[i]] for i in positions],
                [[err_neg[labels[i]] for i in positions], [err_pos[labels[i]] for i in positions]], fmt='o')
        axs.set_xticks(positions)
        axs.set_xticklabels(labels)
    plt.show()







graph({})




