import os
import ast
from itertools import chain
import matplotlib.pyplot as plt
import datetime

base_dir = './LN/LN_results/LN_random_topo'
individual_dir = ['5random_graph_10_1000_19-20/', '10random_graph_10_1000_19-20/', '15random_graph_10_1000_19-20/', '20random_graph_10_1000_19-20/']
entropy_dir = 'graphs/'
data_dir = 'data_files/nodes_affected_per_sample/'

entropy_data = {}
all_frac_captured = {}

os.chdir(base_dir)
for idir in individual_dir:
    os.chdir(idir+entropy_dir)
    entropy_data[idir[:-1]] = []
    for file in os.listdir():
        if file.endswith('.txt'):
            with open(file) as f:
                data = ast.literal_eval(f.read())
            #print(len(data[0]))
            entropy_data[idir[:-1]].extend(list((chain.from_iterable(data))))
    print(len(entropy_data[idir[:-1]]))

    os.chdir('../../')
    os.chdir(idir+data_dir)
    all_frac_captured[idir[:-1]] = []
    for file in os.listdir():
        if file.endswith('.txt'):
            with open(file) as f:
                data = ast.literal_eval(f.read())
            for key, val in data.items():
                all_frac_captured[idir[:-1]].append(val['path_fraction']*100)
    all_frac_captured[idir[:-1]] = sum(all_frac_captured[idir[:-1]])/len(all_frac_captured[idir[:-1]])

    os.chdir('../../../')

labels, data = [*zip(*entropy_data.items())] 
labels = ['5', '10', '15', '20']

colors = []
for i in range(0,len(individual_dir)):
    colors.append('lightgrey')
bp = plt.boxplot(data, patch_artist = True, widths=[0.25 for i in range(0,len(labels))])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
for median in bp['medians']:
    median.set(color ='r', linewidth = 1.5)
for flier in bp['fliers']:
    flier.set(marker='+', alpha = 1, markersize = 4, markeredgecolor='b')

plt.xticks(range(1, len(labels) + 1), labels, weight = 'bold', fontsize=12)
plt.yticks(weight = 'bold', fontsize=12)
ax = plt.gca()
ax2 = ax.twiny()
ax2_labels = [round(all_frac_captured[i],1) for i in all_frac_captured]
ax2.set_xticks(ax.get_xticks())
ax2.set_xbound(ax.get_xbound())
ax2.set_xticklabels(ax2_labels, fontweight='bold', fontsize=12)#, rotation = 20)
ax2.set_xlabel('Transactions Captured (%)', fontsize='x-large', fontweight='bold')
ax.set_xlabel('Average Node Degree', fontsize='x-large', fontweight='bold')
#ax.set_xlabel('Centrality Nodes Selected (with equal cost)', fontsize='large', fontweight='bold')
ax.set_ylabel('Entropy', fontsize='x-large', fontweight='bold')
plt.savefig('LN_random_graph_1000_19-20' + '_' + str(datetime.date.today()) + '_' + \
    str(datetime.datetime.now().minute), dpi=600)
#plt.show()
