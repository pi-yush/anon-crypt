import itertools
import networkx as nx
import random
from random import randint
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from itertools import islice
import sys
import time
import multiprocessing as mp
import json
import datetime
import numpy as np

from numpy import append, int_

def k_shortest_paths(G, source, target, k, weight=None):
    try:
        x = list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    except:
        return None
    return x


def calc_entropy(i,sample_size,N,outdegree,pf,all_entropy,all_frac_captured):
    G = nx.path_graph(N,nx.DiGraph)
    G.add_edge(N-1,0)
    print(len(G.edges))

    entropy = []
    adv_combinations = []

    for j in range(1,sample_size+1):
        temp = []
        while True:
            advn = randint(0,N-1)
            if(advn not in temp):
                temp.append(advn)
            if (len(temp) == i):
                break
        adv_combinations.append(temp)

    adv_combination_entropy = []
    frac_captured = []

    for j in adv_combinations:
        G1 = deepcopy(G)
        rem_edges = []
        for k in j:
            for l in G1.successors(k):
                rem_edges.append([k,l])
        
        G1.remove_edges_from(rem_edges)

        honest_nodes = list(set(G1.nodes) - set(j))
        per_combination_ind_adv_entropy = []
        adv_count = 0
        total_tran_captured = 0

        for adv_node in j:
            adv_count +=1
            prob = []
            norm_factor = 0.0
            entr = 0.0
            tran_count_per_adv = 0

            for node in honest_nodes:
                paths_to_adv = list(nx.all_simple_paths(G1, node, adv_node))
                int_expr = 0.0
                for path in paths_to_adv:
                    #print(path)
                    path_val = 1
                    for index in range(0,len(path)-2):
                        path_val = path_val*(1/outdegree)*pf
                    int_expr+= path_val
                
                if(int_expr != 0.0):
                    prob.append(int_expr)
                    norm_factor += int_expr
                    tran_count_per_adv = tran_count_per_adv + int(int_expr*1000)

            if (norm_factor > 0.0):
                for p in prob:
                    entr = entr + (p/norm_factor)*math.log2(norm_factor/p)
            
            per_combination_ind_adv_entropy += tran_count_per_adv*[entr]
            total_tran_captured += tran_count_per_adv

        for ent in per_combination_ind_adv_entropy:
            adv_combination_entropy.append(ent)

        frac_captured.append((total_tran_captured*100)/(1000*(N-i)))
    entropy.append(adv_combination_entropy)
    all_entropy[N] = adv_combination_entropy
    all_frac_captured[N] = sum(frac_captured)/len(frac_captured)

if __name__ == '__main__':

    manager = mp.Manager()
    all_entropy = manager.dict()
    all_frac_captured = manager.dict()

    N = 1000
    pf = 0.9
    sample_size = 50
    outdegree = 1
    max_hops = 10
    startc = 1000
    endc = 5001
    increment = 1000
    fraction = 0.01
    # adv_nodes = int(N*fraction)

    init = time.time()
    processes = []
    for i in range(startc,endc,increment):
        i = round(i,1)
        adv_nodes = int(i*fraction)
        p = mp.Process(target=calc_entropy,args=(adv_nodes,sample_size,i,outdegree,pf,all_entropy, all_frac_captured))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
    print ('Total time :' + str(time.time()-init))

    print(len(all_entropy))
    #json.dump(all_entropy.copy(),open('Dan_vary_N_' + str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '.txt','w'))
    sorted_dict = dict(sorted(all_entropy.items()))
    labels, data = [*zip(*sorted_dict.items())] 
    labels = [int(i) for i in labels]
    fig = plt.figure()
    ax = fig.add_subplot()

    colors = []
    for i in range(0,round((endc-startc)/increment)+1):
        colors.append('lightgrey')
    bp = plt.boxplot(data, patch_artist = True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color ='r', linewidth = 1.5)
    for flier in bp['fliers']:
        flier.set(marker='+', alpha = 1, markersize = 4, markeredgecolor='b')
    
    plt.xticks(range(1, len(labels) + 1), labels, weight = 'bold', fontsize=12)
    #ax.set_xticks(range(1, len(labels) + 1), labels)
    plt.yticks(range(0,12), weight = 'bold', fontsize=12)
    ax = plt.gca()
    ax.set_xlabel('Total Nodes', fontsize='x-large', fontweight='bold')
    ax.set_ylabel('Entropy', fontsize='x-large', fontweight='bold')
    plt.savefig('Dan_vary_N_scaled' +  '_' + str(datetime.date.today()) + '_' + str(datetime.datetime.now().minute), dpi=600)
    #plt.show()
