import json
import networkx as nx
import os
import pprint
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import csv

def connected_component_subgraphs(G):
	for c in nx.connected_components(G):
		yield G.subgraph(c)

def build_graph_from_snapshot(new, tx_val):

    SNAPSHOT_DIR = './'
    if new:
        SNAPSHOT_FILE = 'channels.json'
    else:
        SNAPSHOT_FILE = 'channels_2018-12-01.json'
    POLICY_FILE = 'policies.json'
    NODE_FILE = 'nodes.json'

    with open(SNAPSHOT_FILE, 'r') as snapshot_file:
        network = json.load(snapshot_file)


    nodemap = {}
    ncount = 0
    nodes_set = set()

    if(os.path.isfile('nodemap.pkl')):
        with open('nodemap.pkl', 'rb') as nmap:
            nodemap = pickle.load(nmap)
    else:
        for edge in network:
            if new:
                if not edge['close']['type']:
                    nodes_set.add(edge['nodes'][0])
                    nodes_set.add(edge['nodes'][1])
            else:  
                if not edge['close_type']:
                    nodes_set.add(edge['node0'])
                    nodes_set.add(edge['node1'])
        
        nodes = list(nodes_set) 
        for n in nodes:
            nodemap[n] = str(ncount)
            ncount += 1

        with open('nodemap.pkl', 'wb') as nmap:
            pickle.dump(nodemap, nmap)
    nodes = list(nodemap.values())

    # To parse data downloaded from a fresh snapshot in May 2021
    if new:
        edges = [(nodemap[edge['nodes'][0]], nodemap[edge['nodes'][1]],
            {
            'weight': 0,
            'short_channel_id': edge['short_channel_id'],
            'direction': '1'
            }
            ) for edge in network if not edge['close']['type'] and edge['satoshis']/2 > tx_val]
    else:
        #To parse historic data downloaded from an old paper
        edges = [(nodemap[edge['node0']], nodemap[edge['node1']],
            {
            'weight': 0,
            'short_channel_id': edge['short_channel_id'],
            'direction': '1'
            }
            ) for edge in network if not edge['close_type'] and edge['satoshis']/2 > tx_val]

    new_edges = deepcopy(edges)
    edge1 = ()
    for edge in new_edges:
        edge1 = (edge[1],edge[0],{'weight':edge[2]['weight'],'short_channel_id': edge[2]['short_channel_id'], 'direction': '0'})
        edges.append(edge1)

    with open(POLICY_FILE, 'r') as policy_file:
        policy = json.load(policy_file)

    count = 0
    ebf = []
    efpm = []
    hashed_policy = {}

    for e in policy:
        if e['short_channel_id'] not in hashed_policy:
            hashed_policy[e['short_channel_id']] = {}
            hashed_policy[e['short_channel_id']]['1'] = {}
            hashed_policy[e['short_channel_id']]['0'] = {}
        hashed_policy[e['short_channel_id']][str(e['direction'])] = e


    with open('hashed_pol.txt', 'w') as hp:
        hp.write(pprint.pformat(hashed_policy))

    weight = []
    for edge in edges:    
        if(edge[2]['short_channel_id'] in hashed_policy):
            #print(hashed_policy[edge[2]['short_channel_id']][edge[2]['direction']])
            if 'base_fee_millisatoshi' in hashed_policy[edge[2]['short_channel_id']][edge[2]['direction']]:
                edge[2]['weight'] = hashed_policy[edge[2]['short_channel_id']][edge[2]['direction']]['base_fee_millisatoshi'] + \
                                    hashed_policy[edge[2]['short_channel_id']][edge[2]['direction']]['delay']*tx_val/1000000000 +\
                                    hashed_policy[edge[2]['short_channel_id']][edge[2]['direction']]['fee_per_millionth']*tx_val/1000000

                # For Histogram
                if (hashed_policy[edge[2]['short_channel_id']][edge[2]['direction']]['base_fee_millisatoshi'] < 5000):
                    weight.append(edge[2]['weight'])
            else:
                edges.remove(edge)    
        else:
            count+=1
            edges.remove(edge)
    print(count)

    G = nx.MultiGraph() 
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    DG = nx.DiGraph() 
    DG.add_nodes_from(nodes)
    DG.add_edges_from(edges)
    #G.add_edges_from(edges_subset)

    print("LN graph created:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges.")

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    print("Components:", len(components), ". Continuing with the largest component.")
    G1 = max(connected_component_subgraphs(G), key=len)
    rem_nodes = list(set(DG.nodes) - set(G1.nodes))
    DG.remove_nodes_from(rem_nodes)
    G.remove_nodes_from(rem_nodes)
    print('Rem node count: ' + str(len(rem_nodes)))
    print(len(DG.nodes), "nodes", len(DG.edges), "edges")
    print(len(G.nodes), "nodes", len(G.edges), "edges")
    print(len(G1.nodes), "nodes", len(G1.edges), "edges")
    return DG, list(DG.nodes)
