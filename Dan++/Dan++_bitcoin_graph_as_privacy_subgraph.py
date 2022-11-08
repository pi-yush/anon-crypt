import itertools
import tracemalloc
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

from numpy import append, int_

def k_shortest_paths(G, source, target, k, weight=None):
    try:
        x = list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    except:
        return None
    return x

N = 1000
C = 6
pf = 0.9
sample_size = 20
outdegree = 8
max_hops = 5
nodes = range(N)
startc = 100
endc = 101
increment = 50


def calc_entropy(i,G,sample_size,N,all_entropy,all_frac_captured):
#for i in range(10,C):
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
        #print(j)
        G1 = deepcopy(G)
        rem_edges = []
        for k in j:
            for l in G1.successors(k):
                rem_edges.append([k,l])
        #print(rem_edges)
        G1.remove_edges_from(rem_edges)

        honest_nodes = list(set(G1.nodes) - set(j))
        per_combination_ind_adv_entropy = []
        adv_count = 0
        pred_count = 0
        frac_captured_per_combination = {}
        total_tran_captured = 0
        for adv_node in j:
            adv_count +=1
            tran_count_per_adv = 0

            #Predecessor extra code
            for pred in G1.predecessors(adv_node):
                prob = []
                prob.append(0.5)
                norm_factor = 0.5
                entr = 0.0
                pred_count += 1
                node_to_pred_count = int(1000/outdegree)

                for node in honest_nodes:
                    paths_to_adv = list(nx.all_simple_paths(G1, node, pred,max_hops))
                   
                    int_expr = 0.0
                    for path in paths_to_adv:
                        #print(path)
                        path_val = 1
                        for index in range(0,len(path)-1):
                            path_val = path_val*(1/outdegree)*pf
                        int_expr+= path_val
                    
                    if(int_expr != 0.0):
                        #To account for thr prob. of a tx reaching the adversary node from the predecessor
                        int_expr = int_expr/outdegree

                        prob.append(int_expr)
                        norm_factor += int_expr
                        if node not in frac_captured_per_combination:
                            frac_captured_per_combination[node] = int_expr*1000
                        else:
                            frac_captured_per_combination[node] += int_expr*1000

                        # Additional code for increasing the entropy points by calculating the
                        # number of clients whose path encounters a particular predecessor node
                        node_to_pred_count = node_to_pred_count + int(int_expr*1000)
                tran_count_per_adv += node_to_pred_count

                if (norm_factor > 0.0):
                    for p in prob:
                        entr = entr + (p/norm_factor)*math.log2(norm_factor/p)
                # per_combination_ind_adv_entropy.append(entr)
                per_combination_ind_adv_entropy += node_to_pred_count* [entr]

            total_tran_captured += tran_count_per_adv
        
        for ent in per_combination_ind_adv_entropy:
            adv_combination_entropy.append(ent)
        
        ## New approach to calculate the fraction of transactions captured
        if (total_tran_captured > 0):
            #print("Transaction: " + str(total_frac_cap))
            frac_captured.append((total_tran_captured*100)/(1000*(N-i)))
            
    entropy.append(adv_combination_entropy)
    
    if (i == 1):
        all_entropy[i] = adv_combination_entropy
    else:
        all_entropy[int((i*100)/N)] = adv_combination_entropy
        all_frac_captured[int((i*100)/N)] = sum(frac_captured)/len(frac_captured)
    

if __name__ == '__main__':
    manager = mp.Manager()
    all_entropy = manager.dict()
    all_frac_captured = manager.dict()
    MG = nx.random_k_out_graph(N,outdegree,1000,False)
    print(len(MG.edges))
    new_edges = []
    for edge in MG.edges:
        nedge = []
        if(edge[2] == 1):
            nedge.append(edge[0])
            while(True):
                i = randint(0,N)
                if(i != edge[1]):
                    nedge.append(i)
                    break
            new_edges.append(nedge)
    print(new_edges)

    G = nx.DiGraph(MG)
    G.add_edges_from(new_edges)
    print(len(G.edges))

    init = time.time()
    processes = []
    xvalues = []
    for i in range(startc,endc,increment):
        p = mp.Process(target=calc_entropy,args=(i,G,sample_size,N,all_entropy,all_frac_captured))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
    print ('Total time :' + str(time.time()-init))
    print(len(all_entropy))
    json.dump(all_entropy.copy(),open('Dan++_vary_adversary_fraction' + str(N) + '_' + str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '.txt','w'))
    sorted_dict = dict(sorted(all_entropy.items()))
    labels, data = [*zip(*sorted_dict.items())] 
    labels = [int(i) for i in labels]
    #plt.show()
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
    
    plt.xticks(range(1, len(labels) + 1), labels, weight = 'bold', fontsize=9)
    plt.yticks(weight = 'bold')
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2_labels = [round(all_frac_captured[i],1) for i in sorted(all_frac_captured)]
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    #ax2.set_xticks(ax2_labels)
    ax2.set_xticklabels(ax2_labels, fontweight='bold')
    ax2.set_xlabel('Transactions Captured (%)', fontsize='large', fontweight='bold')
    ax.set_xlabel('Colluding Nodes (%)', fontsize='large', fontweight='bold')
    ax.set_ylabel('Entropy', fontsize='large', fontweight='bold')
    plt.savefig('Dan++_Bitcoin_p2p_graph_as_privacy_subgraph_pred_scaled' + '_' + str(datetime.date.today()) + '_' + str(datetime.datetime.now().minute))
    #plt.show()
