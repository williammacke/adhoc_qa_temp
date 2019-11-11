import matplotlib.pyplot as plt

data = []
f = open('figure3_results.dat' ,'r')
for l in f:
    data += [[float(i) for i in l.split()]]
f.close()


avg = [sum(x)/len(x) for x in data]
err_pos = []
err_neg = []
for qt, mu in zip(data, avg):
    err_p = []
    err_n = []
    for d in qt:
        if d > mu:
            err_p += [d-mu]
        else:
            err_n += [mu-d]
    if len(err_p):
        err_pos += [sum(err_p)/len(err_p)]
    else:
        err_pos += [0]
    if len(err_n):
        err_neg += [sum(err_n) / len(err_n)]
    else:
        err_neg += [0]


labels = ['X', '1', 'ZQ', 'Random']
positions = list(range(4))

fig, ax = plt.subplots(1, 1)

ax.errorbar(positions, avg, [err_neg, err_pos], fmt='o', capsize=10)
ax.set_xticks(positions)
ax.set_xticklabels(labels)
plt.show()
