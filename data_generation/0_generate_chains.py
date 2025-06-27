import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import pickle
import argparse
import copy
import os

# add command line arguments
parser = argparse.ArgumentParser(description='Generate chains with dependencies.')
parser.add_argument('--n', type=int, default=8, help='Number of chains.')
parser.add_argument('--m', type=int, default=12, help='Length of chains.')
parser.add_argument('--max_rules', type=int, default=2, help='Maximum number of rules.')
parser.add_argument('--cross_ratio1', type=float, default=0.5, help='Crossbreed ratio for the first automata.')
parser.add_argument('--indexing_margin', type=int, default=0, help='Indexing margin for nodes.')
args = parser.parse_args()

n = args.n
m = args.m
max_rules = args.max_rules
indexing_margin = args.indexing_margin

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")

def gen_chains(n=n, m=m):
    chains = defaultdict(list)

    for i in range(n):
        G = nx.Graph()
        nodes = [(j+indexing_margin) for j in range(m)]
        edges = list(zip(nodes[:-1], nodes[1:]))

        G.add_edges_from(edges)
        chains[i] = G

    return chains

def add_enabling(chains):
    n = len(chains)
    
    for chain_id in range(n-1):
        G = chains[chain_id]
        
        for edge in G.edges():
            if random.random() < 0.5:
                num_deps = random.randint(0, max_rules)
                
                if num_deps > 0:
                    later_chains = list(range(chain_id + 1, n))
                    selected_chains = random.sample(later_chains, min(num_deps, len(later_chains)))
                    
                    enabling_conditions = []
                    for enabling_chain in selected_chains:
                        enabling_G = chains[enabling_chain]
                        enabling_node = random.choice(list(enabling_G.nodes()))
                        enabling_conditions.append({'automata_id':enabling_chain,'node_id':enabling_node})
                        
                    enabling_conditions_sorted = sorted(enabling_conditions, key=lambda x: x['automata_id'])
                    
                    G.edges[edge]['enabling'] = enabling_conditions_sorted
    
    return chains

chains = add_enabling(gen_chains())
chains2 = add_enabling(gen_chains())

def add_list_to_edges(graph):
    for chain_id in range(len(graph)):
        G = graph[chain_id]
        for edge in G.edges(data=True):
            if 'enabling' in edge[2]:
                edge[2]['enabling'] = [edge[2]['enabling'], {'automata_id':n-1,'node_id':9999}]
                

def update_chain_rules(chain, num):
    for i in range(len(chain)):
        for edge in chain[i].edges(data=True):
            if 'enabling' in edge[2]:
                for enabling in edge[2]['enabling']:
                    enabling['automata_id'] = enabling['automata_id'] - num

def get_automata_with_dependent_chains(num):
    found = False
    while not found:
        chain = add_enabling(gen_chains())
        if num <= 1:
            return chain
        
        for i in range(num, 1, -1):
            for edge in chain[n-i].edges(data=True):
                if 'enabling' in edge[2]:
                    found = True
                    break
                else:
                    found = False
        
        if found:
            return chain

def crossbreed(chain1, chain2, chain1_ratio=0.5):
    new_automata = add_enabling(gen_chains())

    res_graph = defaultdict(list)
    res_graph[0] = new_automata[0]
    # res_graph[1] = chain1[n-4]
    # res_graph[2] = chain1[n-3]
    # res_graph[3] = chain1[n-2]
    # res_graph[4] = chain1[n-1]
    # res_graph[5] = chain2[n-3]
    # res_graph[6] = chain2[n-2]
    # res_graph[7] = chain2[n-1]

    chains = [chain1, chain2]

    counter = int((n-1) * chain1_ratio)
    initial_counter = counter
    # print(f"Initial counter: {initial_counter}, Counter: {counter}")
    chain_num = 0
    next_automata_id_start = 0
    for i in range(1, n):
        if counter == 0:
            chain_num = 1
            counter = (n-1) - initial_counter
            next_automata_id_start = i
        res_graph[i] = chains[chain_num][n - counter]
        # print(f"res_graph[{i}] = chains[{chain_num}][{n  - counter}]")
        counter -= 1
    
    random_edge1 = random.randint((0+indexing_margin), (m-2+indexing_margin))
    while True:
        random_edge2 = random.randint((0+indexing_margin), (m-2+indexing_margin))
        if random_edge1 != random_edge2:
            break

    res_graph[0].edges[random_edge1, random_edge1+1]['enabling'] = [{'automata_id': 1, 'node_id': random.randint((0+indexing_margin), (m-1+indexing_margin))}]
    res_graph[0].edges[random_edge2, random_edge2+1]['enabling'] = [{'automata_id': next_automata_id_start, 'node_id': random.randint((0+indexing_margin), (m-1+indexing_margin))}]

    # add_list_to_edges(new_automata)

    return res_graph

chain1_ratio = args.cross_ratio1

automata1 = get_automata_with_dependent_chains(num=int((n-1)*chain1_ratio))
automata1_cpy = copy.deepcopy(automata1)
add_list_to_edges(automata1_cpy)

automata2 = get_automata_with_dependent_chains(num=(n-1) - int((n-1)*chain1_ratio))
automata2_cpy = copy.deepcopy(automata2)
add_list_to_edges(automata2_cpy)

update_chain_rules(automata1, num=abs((n-1) -int((n-1) * chain1_ratio)))
update_chain_rules(automata2, num=0)


res_G = crossbreed(automata1, automata2, chain1_ratio)


add_list_to_edges(res_G)

with open(f'{data_dir}/automata1.pkl', 'wb') as f:
    pickle.dump(automata1_cpy, f)

with open(f'{data_dir}/automata2.pkl', 'wb') as f:
    pickle.dump(automata2_cpy, f)

with open(f'{data_dir}/res_G.pkl', 'wb') as f:
    pickle.dump(res_G, f)


