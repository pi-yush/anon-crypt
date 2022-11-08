import networkx as nx
import matplotlib.pyplot as plt
import pprint
import math
import copy
from random import randint
import pickle
import os.path
import datetime
import json
import numpy as np
import multiprocessing as mp
import time
import os
import sys

#from LN_snapshot_to_graph_converter import build_graph_from_snapshot
from LN_snap_to_digraph_converter_tx_value_cltv import build_graph_from_snapshot

data_dir = 'LN_results/'
files_dir = 'data_files/'
graph_dir = 'graphs/'

#for size in range(0,sample_size):
def calc_entropy(size, G, adv_nodes_count, nodes_affected_overall, entropy, path, node_list, nodes, int_dir):
    adv_node = []
    for i in range(0,adv_nodes_count):
        adv_node.append(node_deg[i][0])
        print('Node:' + str(node_deg[i][0]) + ' Degree:' + str(G.degree[node_deg[i][0]]))

    # Dictionaries containing the adversary node(s) as key and their sub-paths (predecessor and successor) as values
    b = {}
    b_copy={}

    opt_paths = {} # Contains adversary nodes as key and a list of paths that pass through it as values
    i=0
    total_paths = 0

    #Extract the adversaries as well as the sub-paths captured by those adversaries including the 
    # predecessor and successor from which the transaction will be received.
    #For each path, start from the leftmost-1 and rightmost-1 hop, and as soon as an
    #adversary node is encountered, save the sub-path <predecessor--adversary_node--successor> 
    # to the list of sub-paths for this adversary.
    #If an adversary node falls in both the right and left part of the path, discard all the nodes
    # between them and add the new adversary combination in opt_paths.
    for key in path:
        for k, v in path[key].items():
            total_paths += 1
            if len(v) <= 2:
                continue
            left = 1
            right = int(len(v))-2
            if(int(len(v))%2 == 0):
                mid = int(len(v)/2) - 1
            else:
                mid = int(len(v)/2)
            left_flag= False
            right_flag=False

            while(left<=mid):
                if (str(v[left]) in adv_node):
                    if(len(v) == 3):
                        if(str(v[left]) not in b_copy):
                            b_copy[str(v[left])]=set()  
                        b_copy[str(v[left])].add('-'.join(str(e) for e in v))
                    else:
                        if(str(v[left]) not in b_copy):
                            b_copy[str(v[left])]=set()
                            
                        b_copy[str(v[left])].add(str(v[left-1]) + '-' + str(v[left]) + '-' + str(v[left+1]))
                        
                    left_flag =True
                    break
                left+=1
            
            while (right>mid and len(v)>3):
                if(str(v[right]) in adv_node):
                    if(str(v[right]) not in b_copy):
                        b_copy[str(v[right])]=set()
                        
                    b_copy[str(v[right])].add(str(v[right-1]) + '-' + str(v[right]) + '-' + str(v[right+1]))
                    right_flag = True
                    break
                right-=1

            if (left_flag and right_flag):
                    pkey = str(v[left]) + '-' + str(v[right])
                    temp = v[left-1:right+2]
                    temp = '-'.join(str(node) for node in temp)
                    if pkey not in b_copy:
                        b_copy[pkey]=set()
                    
                    b_copy[pkey].add(temp)                

                    if pkey not in opt_paths:
                        opt_paths[pkey] = []
                    opt_paths[pkey].append(v)
            if left_flag and not right_flag:
                if str(v[left]) not in opt_paths:
                    opt_paths[str(v[left])] = []
                opt_paths[str(v[left])].append(v)
            if right_flag and not left_flag:
                if str(v[right]) not in opt_paths:
                    opt_paths[str(v[right])] = []
                opt_paths[str(v[right])].append(v)

    #Copying all the processed paths to 'b'
    for key in b_copy:
        b[key] = list(b_copy[key])

    a = {} # Contains the list of paths (as strings) passing through adversary node.
    nodes_affected = {}
    # Variables to capture overall affected nodes and paths
    nodes_affected_overall[size] = {}
    nao = nodes_affected_overall[size]
    nao['src_node_list'] = set() 
    nao['path_count'] = 0
    nao['all_path_count'] = 0
    nao['dst_node_list'] = set()

    # Converting paths to strings
    for key in opt_paths:
        for v in opt_paths[key]:
            if key not in nodes_affected:
                nodes_affected[key] = {}
                nodes_affected[key]['src_node_list'] = set()
                nodes_affected[key]['dst_node_list'] = set()
                nodes_affected[key]['path_count'] = 0
                nodes_affected[key]['src_node_count'] = 0
                nodes_affected[key]['dst_node_count'] = 0
                nodes_affected[key]['path_fraction'] = 0
            nodes_affected[key]['src_node_list'].add(str(v[0]))
            nodes_affected[key]['dst_node_list'].add(str(v[-1]))
            nodes_affected[key]['path_count']+=1
            nao['path_count'] += 1
            path_str = '-'.join(str(e) for e in v)
            if (key not in a):
                a[key] = []
            a[key].append(path_str)

        nodes_affected[key]['src_node_count'] = len(nodes_affected[key]['src_node_list'])
        nodes_affected[key]['dst_node_count'] = len(nodes_affected[key]['dst_node_list'])
        nodes_affected[key]['path_fraction'] = nodes_affected[key]['path_count']/total_paths

        #Updating overall sample data
        nao['src_node_list'].update(nodes_affected[key]['src_node_list'])
        nao['dst_node_list'].update(nodes_affected[key]['dst_node_list'])
        nao['all_path_count'] += nodes_affected[key]['path_count']

        #Emptying individual sample src and dst sets to avoid clutter in the saved txt file
        nodes_affected[key].pop('src_node_list')
        nodes_affected[key].pop('dst_node_list')

    nao['src_node_count'] = len(nao['src_node_list'])
    nao['dst_node_count'] = len(nao['dst_node_list'])
    nao['total_paths'] = total_paths
    nao['path_fraction'] = nao['path_count']/total_paths

    #Emptying individual sample src and dst sets to avoid clutter in the saved txt file
    nao.pop('src_node_list')
    nao.pop('dst_node_list')
    nodes_affected_overall[size] = nao

    if not os.path.isdir(data_dir + int_dir + files_dir + 'ind_nodes_affected/'):
        os.makedirs(data_dir + int_dir + files_dir + 'ind_nodes_affected/')
    with open(data_dir + int_dir + files_dir + 'ind_nodes_affected/nodes_affected_' + str(size) + '_' + \
        str(datetime.date.today()) + '_' + str(int(time.time())) + '.txt', 'w') as f:
        f.write(pprint.pformat(nodes_affected))

    #A dictionary that holds all the path for a particular client as originator, and passing
    # through an adversary node as intermediary.
    # adv_paths[x][y] will contain list of paths that originate from node 'x' and pass through adversary node 'y' 
    adv_paths = {}
    for key in a:
        for val in a[key]:
            x = val.split('-')[0]
            if x not in adv_paths:
                adv_paths[x] = {}
            if key not in adv_paths[x]:
                adv_paths[x][key] = []
            adv_paths[x][key].append(val)

    # Uncomment to get detailed data for further analysis
    '''if not os.path.isdir(data_dir + int_dir + files_dir + 'honest_to_adv_paths/'):
        os.makedirs(data_dir + int_dir + files_dir + 'honest_to_adv_paths/')
    with open(data_dir + int_dir + files_dir + 'honest_to_adv_paths/' + str(size) + '_' + str(int(time.time())) + '.txt', 'w') as f:
        f.write(pprint.pformat(adv_paths))

    if not os.path.isdir(data_dir + int_dir + files_dir + 'all_adv_paths/'):
        os.makedirs(data_dir + int_dir + files_dir + 'all_adv_paths/')
    with open(data_dir + int_dir + files_dir + 'all_adv_paths/' + str(size) + '_' + str(int(time.time())) + '.txt', 'w') as f:
        f.write(pprint.pformat(a))
    #pprint.pprint(b)
    if not os.path.isdir(data_dir + int_dir + files_dir + 'subpaths/'):
        os.makedirs(data_dir + int_dir + files_dir + 'subpaths/')
    with open(data_dir + int_dir + files_dir + 'subpaths/' + str(size) + '_' + str(int(time.time())) + '.txt', 'w') as f:
        f.write(str(b))'''

    #Parallezing the processing to calculate the probability distribution of originators for each adversary.
    # For each adversary node(s) we invoke the entropy calculating funtion
    processes = []
    for k in b:
        p = mp.Process(target=entropy_per_adv_node,args=(k,b,adv_paths,nodes,int_dir))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()

def entropy_per_adv_node(k,b,adv_paths,nodes,interval_dir):
        prob = {}
        entr = []
        #For each subpath of a particular adversary, calculate the probability distributions of each node
        #being the potential originator.
        
        for subpath in b[k]:
            norm_factor = 0.0
            entropy_multiplier = 0
            for key in adv_paths:
                count = 0
                # Temporarily prob[key] contain P[Bjkl|Ai]
                # Count the number of paths of a particular node that pass through the adversary subpath and divide that
                # by the total  number of paths originating from that node.
                if(k in adv_paths[key]):
                    for v in adv_paths[key][k]:
                        if (v.find(subpath) != -1):
                            count = count + 1
                    prob[key] = float(count)/float(nodes - 1) 
                    #Summation(P[Bjkl|Ai]) for all i
                    norm_factor = norm_factor + prob[key]
                #else:
                #    prob[key] = 0.0
                entropy_multiplier += count
            
            if (norm_factor != 0):            
                for key in adv_paths:
                    # P[Ai|Bjkl] = P(Ai)*P[Bjkl|Ai]/P[Bjkl]
                    # equivalant to P[Bjkl|Ai]/Summation(P[Bjkl|Ai])
                    if key in prob:
                        prob[key] = prob[key]/norm_factor                 
                #print(prob)
                ent = 0.0
                for p, val in prob.items():
                    if (val != 0):
                        ent += (-val*math.log2(val))
                #entr.append(ent)
                entr += entropy_multiplier*[ent]
                
        # Save the entropy data for processing and plotting graphs seperately
        if not os.path.isdir(data_dir + interval_dir + graph_dir + 'ind_entr_data/'):
            os.makedirs(data_dir + interval_dir + graph_dir + 'ind_entr_data/')
        with open(data_dir + interval_dir + graph_dir + 'ind_entr_data/LN_entropy' +\
                '_' + str(k) + '_' + str(datetime.date.today()) + '_' + str(int(time.time())) + '.txt','w') as f:
            json.dump(entr,f)
        
        #Uncomment to obtian entropy graphs for each adversary subpath.
        '''fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('Node Number', fontsize='large', fontweight='bold')
        ax.set_ylabel('Entropy', fontsize='large', fontweight='bold')

        colors = []
        for i in range(0,1):
            colors.append('lightgrey')
        bp = plt.boxplot(entr, patch_artist = True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        for median in bp['medians']:
            median.set(color ='r', linewidth = 1.5)
        for flier in bp['fliers']:
            flier.set(marker='+', alpha = 1, markersize = 4, markeredgecolor='b')
        labels = [k]
        plt.xticks(range(1, len(labels) + 1), labels, weight = 'bold', fontsize=9)
        plt.yticks(weight = 'bold')
        if not os.path.isdir(data_dir + interval_dir + graph_dir + 'ind_graphs/'):
            os.makedirs(data_dir + interval_dir + graph_dir + 'ind_graphs/')
        plt.savefig(data_dir + interval_dir + graph_dir + 'ind_graphs/LN_vary_adv_frac' + '_' + str(k) + '_' + \
            str(datetime.datetime.now().minute))'''
        return

if __name__ == '__main__':

    if len(sys.argv) != 4:
        #For most evaluations sample_size = 1
        #Adversary_node_count values should be 12, 30, 60 and 120 corresponding to 1, 2.5, 5 and 10 % of the 2018
        # topology which is about 1200 nodes in size.
        print ('filename.py <tx_value> <sample_size> <adversary_node_count>') 
        sys.exit()

    entropy = []
    nodes_affected_overall = {}

    init = time.time()

    tx_val = int(sys.argv[1])
    G,node_list = build_graph_from_snapshot(False, tx_val)

    nodes = len(node_list)
    print(nodes)

    # Finding node degrees:
    node_deg = sorted(G.degree, key=lambda x: x[1], reverse=True) 
    
    # Calculating Dijkstra paths and saving them so that multiple runs for the same snapshot are much faster
    dj_paths = 'dj_' + str(tx_val) + '.pkl'
    print('Calculating the shortest paths!!')
    if(os.path.isfile(dj_paths)):
        with open(dj_paths, 'rb') as dj_paths:
            path = pickle.load(dj_paths)
    else:
        path = dict(nx.all_pairs_dijkstra_path(G))
        with open(dj_paths, 'wb') as dj_paths:
            pickle.dump(path,dj_paths)

    print('Calculated the shortest paths!')

    sample_size = int(sys.argv[2])
    adv_nodes_count = int(sys.argv[3])
    interval_dir = 'top_degree' + '_' + str(adv_nodes_count) + '_' + str(tx_val) + '/'

    calc_entropy(0,G,adv_nodes_count,nodes_affected_overall,entropy,\
             path,node_list,nodes,interval_dir)
    print ('Total time :' + str(time.time()-init))

    if not os.path.isdir(data_dir + interval_dir + files_dir + 'nodes_affected_per_sample/'):
        os.makedirs(data_dir + interval_dir + files_dir + 'nodes_affected_per_sample/')
    with open(data_dir + interval_dir + files_dir +'nodes_affected_per_sample/' + str(int(time.time())) + '.txt', 'w') as f:
        f.write(pprint.pformat(nodes_affected_overall.copy()))

    #Uncomment for saving graphs of this run
    '''fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Sample Number', fontsize='large', fontweight='bold')
    ax.set_ylabel('Entropy', fontsize='large', fontweight='bold')

    colors = []
    for i in range(0,len(entropy)):
        colors.append('lightgrey')
    bp = plt.boxplot(entropy, patch_artist = True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color ='r', linewidth = 1.5)
    for flier in bp['fliers']:
        flier.set(marker='+', alpha = 1, markersize = 4, markeredgecolor='b')
    labels = [i for i in range(1,len(entropy)+1)]
    plt.xticks(range(1, len(labels) + 1), labels, weight = 'bold', fontsize=9)
    plt.yticks(weight = 'bold')
    if not os.path.isdir(data_dir + interval_dir + graph_dir):
        os.makedirs(data_dir + interval_dir + graph_dir)
    plt.savefig(data_dir + interval_dir + graph_dir + 'LN_vary_adv_frac' + '_' + str(nodes) + '_' + \
        str(adv_nodes_count) + '_' + str(sample_size) + '_' + str(datetime.datetime.now().minute))'''
