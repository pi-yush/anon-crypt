import os
import ast
from itertools import chain
import matplotlib.pyplot as plt
import datetime

base_dir = '/home/piyushs/LN/longitudinal/LN_results' #Replace with your own base directory
individual_dir =['top_degree_12/', 'channels_2019-07-01top_degree_39/', 'channels_2020-02-01top_degree_52/', 'top_degree_80_2/'] #Point to appropriate directories
entropy_dir = 'graphs/ind_entr_data'
data_dir = 'data_files/nodes_affected_per_sample/'

entropy_data = {}
all_frac_captured = {}

os.chdir(base_dir)
for idir in individual_dir:
    # For 2018 topology, the directory structure is a bit different, so processed seperately.
    if(idir.find('_12') != -1):
        os.chdir(idir+'graphs')
    else:
        os.chdir(idir+entropy_dir)
    
    entropy_data[idir[:-1]] = []
    for file in os.listdir():
        count = 0
        if file.endswith('.txt'):
            count +=1
            with open(file) as f:
                data = ast.literal_eval(f.read())
                
            if(idir.find('_12') != -1):
                entropy_data[idir[:-1]].extend(list((chain.from_iterable(data))))
            else:
                entropy_data[idir[:-1]].extend(list(data))
        if (count == 10):
            break
    print(len(entropy_data[idir[:-1]]))

    if(idir.find('_12') != -1):
        os.chdir('../../')
    else:
        os.chdir('../../../')
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
labels = ['2018', '2019', '2020', '2021'] 

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
#ax2.set_xticks(ax2_labels)
ax2.set_xticklabels(ax2_labels, fontweight='bold', fontsize=12)#, rotation = 20)
ax2.set_xlabel('Transactions Captured (%)', fontsize='x-large', fontweight='bold')

ax.set_ylabel('Entropy', fontsize='x-large', fontweight='bold')
plt.savefig('LN_longitudinal_top_degree_percentage_cltv' + '_' + str(datetime.date.today()) + '_' + \
    str(datetime.datetime.now().minute), dpi=600)
#plt.show()
