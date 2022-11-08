import os
import ast
from itertools import chain
import matplotlib.pyplot as plt
import datetime

base_dir = './LN/LN_results/'
#individual_dir = ['top_degree_12/', 'top_degree_30/', 'top_degree_60/', 'top_degree_120/']
individual_dir = ['top_degree_60_cltv/', 'best_k_5__top_60/']
entropy_dir = 'graphs/'
data_dir = 'data_files/nodes_affected_per_sample/'

entropy_data = {}
all_frac_captured = {}

os.chdir(base_dir)
for idir in individual_dir:
    os.chdir(idir+entropy_dir)
    entropy_data[idir[:-1]] = []
    for file in os.listdir():
        count = 0
        if file.endswith('.txt'):
            count +=1
            with open(file) as f:
                data = ast.literal_eval(f.read())
            entropy_data[idir[:-1]].extend(list((chain.from_iterable(data))))
        if (count == 10):
            break
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
#labels = ['1', '2.5', '5', '10']
labels = ['k=1', 'k=5']

#labels = ['5', '10', '15', '20']
#labels = ['low-middle', 'middle-high', 'high', 'top-1']
#labels = ['1-5']

colors = []
for i in range(0,len(individual_dir)):
    colors.append('lightgrey')
bp = plt.boxplot(data, patch_artist = True, widths=[0.14 for i in range(0,len(labels))])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
for median in bp['medians']:
    median.set(color ='r', linewidth = 1.5)
for flier in bp['fliers']:
    flier.set(marker='+', alpha = 1, markersize = 4, markeredgecolor='b')

plt.xticks(range(1, len(labels) + 1), labels, weight = 'bold', fontsize=12)
plt.yticks(weight = 'bold', fontsize=12)
ax = plt.gca()

# Uncomment for plotting the transactions captured
# ax2 = ax.twiny()
# ax2_labels = [round(all_frac_captured[i],1) for i in all_frac_captured]
# ax2.set_xticks(ax.get_xticks())
# ax2.set_xbound(ax.get_xbound())
# ax2.set_xticklabels(ax2_labels, fontweight='bold', fontsize=12)#, rotation = 20)
# ax2.set_xlabel('Transactions Captured (%)', fontsize='x-large', fontweight='bold')

ax.set_ylabel('Entropy', fontsize='x-large', fontweight='bold')
#plt.savefig('LN_1200_top_degree_percentage' + '_' + str(datetime.date.today()) + '_' + \
#   str(datetime.datetime.now().minute), dpi=600)
plt.savefig('LN_1200_top_degree_best-k' + '_' + str(datetime.date.today()) + '_' + \
   str(datetime.datetime.now().minute), dpi=600)
#plt.show()
