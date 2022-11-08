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

data_dir = 'LN_results/'
files_dir = 'data_files/'
graph_dir = 'graphs/'

def calc_entropy(size, G, adv_nodes_count, nodes_affected_overall, entropy, path, node_list, nodes, \
    int_dir, node_deg):
    adv_node = []
    for i in range(0,adv_nodes_count):
        adv_node.append(node_deg[i][0])
        print('Node:' + str(node_deg[i][0]) + ' Degree:' + str(G.degree[node_deg[i][0]]))

    b = {}
    b_copy={}

    opt_paths = {}
    i=0
    total_paths = 0
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

    for key in b_copy:
        b[key] = list(b_copy[key])

    a = {}
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

    adv_paths = {}
    for key in a:
        for val in a[key]:
            x = val.split('-')[0]
            if x not in adv_paths:
                adv_paths[x] = {}
            if key not in adv_paths[x]:
                adv_paths[x][key] = []
            adv_paths[x][key].append(val)

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
        for subpath in b[k]:
            norm_factor = 0.0 
            entropy_multiplier = 0
            for key in adv_paths:
                count = 0
                # Temporarily prob[key] contain P[Bjkl|Ai]
                if(k in adv_paths[key]):
                    for v in adv_paths[key][k]:
                        if (v.find(subpath) != -1):
                            count = count + 1
                    prob[key] = float(count)/float(nodes - 1) 
                    #Summation(P[Bjkl|Ai]) for all i
                    norm_factor = norm_factor + prob[key]
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

                entr += entropy_multiplier*[ent]

        if not os.path.isdir(data_dir + interval_dir + graph_dir + 'ind_entr_data/'):
            os.makedirs(data_dir + interval_dir + graph_dir + 'ind_entr_data/')
        json.dump(entr,open(data_dir + interval_dir + graph_dir + 'ind_entr_data/LN_entropy' +\
            '_' + str(k) + '_' + str(datetime.date.today()) + '_' + str(int(time.time())) + '.txt','w'))
        
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
            str(datetime.datetime.now().minute))
        '''


if __name__ == '__main__':
    entropy = []
    nodes_affected_overall = {}

    init = time.time()

    nodes = int(sys.argv[1])
    factor = int(sys.argv[2])
    deg_factor=float(factor)/nodes # Used in Graph constrction
    if(os.path.isfile(str(factor) + '_' + str(nodes) + 'rg.pkl')):
        with open(str(factor) + '_' + str(nodes) + 'rg.pkl', 'rb') as dj_paths:
            G = pickle.load(dj_paths)
    else:
        G = nx.gnp_random_graph(nodes, deg_factor, 12345)
        nx.set_edge_attributes(G, {e: {'weight': randint(int(sys.argv[3]), int(sys.argv[4]))} for e in G.edges}) 
        with open(str(factor) + '_' + str(nodes) +'rg.pkl', 'wb') as dj_paths:
            pickle.dump(G,dj_paths)

    #nodes = len(node_list)
    print(nodes)

    # Finding node degrees:
    node_deg = sorted(G.degree, key=lambda x: x[1], reverse=True) 
    with open(str(factor) + '_' + str(nodes) + 'random_graph_degree.txt', 'w') as f:
        f.write(str(node_deg))

    #bw_centrality = {}
    if(os.path.isfile(str(factor) + '_' + str(nodes) + 'random_graph_centrality.pkl')):
        with open(str(factor) + '_' + str(nodes) + 'random_graph_centrality.pkl', 'rb') as dj_paths:
            bw_centrality = pickle.load(dj_paths)
    else:
        #Finding betweenness centrality or path frequency
        bw_c = nx.betweenness_centrality(G, normalized=False, weight='weight')
        bw_centrality = sorted(bw_c.items(), key=lambda x: x[1], reverse=True)
        # bw_centrality = nx.betweenness_centrality(G, normalized=False, weight='weight') 
        with open(str(factor) + '_' + str(nodes) + 'random_graph_betwenenss.txt', 'w') as f:
            f.write(str(bw_centrality))        
        with open(str(factor) + '_' + str(nodes) + 'random_graph_centrality.pkl', 'wb') as dj_paths:
            pickle.dump(bw_centrality,dj_paths)

    bwc = []
    bwc_list = {}
    co = 0
    for bw in bw_centrality:
        if (bw[1] > 1.0):
            bwc.append(bw[1])
            bwc_list[bw[0]] = bw[1]
            co+=1
    #print(co)
    
    sorted_bw_list = sorted(bwc_list.items(), key=lambda x: x[1], reverse=True)
    i = 0
    j = 0
    interval_list = {}
    total_list = []
    boundary = int(len(sorted_bw_list)/4)
    for key, val in sorted_bw_list:
        total_list.append(key)
        if(i % boundary == 0):
            j = i
            interval_list[j] = []
            interval_list[j].append(key)
        else:
            interval_list[j].append(key)
        i+=1
        if (i == 100):
            break

    print('Calculating the shortest paths!!')
    if(os.path.isfile(str(factor) + '_' + str(nodes) + 'random_graph_dj.pkl')):
        with open(str(factor) + '_' + str(nodes) + 'random_graph_dj.pkl', 'rb') as dj_paths:
            path = pickle.load(dj_paths)
    else:
        path = dict(nx.all_pairs_dijkstra_path(G))
        with open(str(factor) + '_' + str(nodes) + 'random_graph_dj.pkl', 'wb') as dj_paths:
            pickle.dump(path,dj_paths)

    print('Calculated the shortest paths!')
    sample_size = 1

    adv_nodes_count = 10
    ld = 1
    hd = 5
    interval = 75
    interval_dir = str(factor) + 'random_graph' + '_' + str(adv_nodes_count) + '_' + str(nodes) + '_' + sys.argv[3] + '-' + sys.argv[4]  + '/'
    calc_entropy(0,G,adv_nodes_count,nodes_affected_overall,entropy,\
             path,nodes,ld,hd,interval_dir,total_list)
    print ('Total time :' + str(time.time()-init))

    if not os.path.isdir(data_dir + interval_dir + files_dir + 'nodes_affected_per_sample/'):
        os.makedirs(data_dir + interval_dir + files_dir + 'nodes_affected_per_sample/')
    with open(data_dir + interval_dir + files_dir +'nodes_affected_per_sample/' + str(int(time.time())) + '.txt', 'w') as f:
        f.write(pprint.pformat(nodes_affected_overall.copy()))

    if not os.path.isdir(data_dir + interval_dir + graph_dir):
        os.makedirs(data_dir + interval_dir + graph_dir)
    json.dump(entropy,open(data_dir + interval_dir + graph_dir +'LN_entropy' +\
         '_' + str(nodes) + '_' + str(adv_nodes_count) + '_' + str(sample_size) + '_' + str(datetime.date.today()) +\
              '_' + str(int(time.time())) + '.txt','w'))
    #new_ent = json.load(open('Dan++_vary_adversary_fraction' + str(N) + '_' + str(datetime.now()) + '.txt'))
    fig = plt.figure()
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
        str(adv_nodes_count) + '_' + str(sample_size) + '_' + str(datetime.datetime.now().minute))





