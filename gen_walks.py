import math
import time
import networkx as nx
import node2vec
import time
from gensim.models import Word2Vec
import numpy as np

walks_per_graph=300
walk_length=10
p=1.0
q=1.0
dimensions=50
window_vector_size=10
iter=5
worker=8
pseudo_count=0.01
def Graph(edge_path):
    edges = []
    with open(edge_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            edges.append(line)
        G = nx.DiGraph()
    for edge in edges:
        if edge[2]=='None':
            edge[2]=0.00895
        G.add_edge(edge[0], edge[1], weight=edge[2])
    count = 1
    for node1 in G.nodes():
        lia=0
        G.nodes()[node1]['D'] = {}
        G.nodes()[node1]['L'] = 0
        ava = []
        for node2 in G.nodes():
            if nx.has_path(G,node1, node2) and node1 != node2:
                ava.append(node2)
                paths1 = nx.all_simple_paths(G, node1, node2)
                paths2 = nx.all_simple_paths(G, node1, node2)
                G.nodes[node1]['D'][node2] = calculate_EID(G, paths1)
                lia+=calculate_LIA(G, paths2)

        count += 1
        G.nodes[node1]['L'] =lia
        G.nodes[node1]['reachable_nodes'] = ava
    return G

def calculate_EID(graph, paths):
    """
        EID: effective infection distance
        D(u, v) = min{1 − log2(P(u, v))}
        P(u, v) = P(u, l1) × P(l1, l2) × ... × P(ln−1, ln) × P(ln, v)
        :return:D(u, v)
    """
    pro_list = []
    for path in paths:
        pro = 1
        for i in range(len(path)):
            if i != len(path) - 1:
                p = float(graph[path[i]][path[i + 1]]['weight']) / graph.out_degree[path[i]]
                pro = pro * p
        pro = 1 - math.log(pro, 2)
        pro_list.append(pro)
    return min(pro_list)

def calculate_LIA(graph, paths):
    """
        LIA: local infecting ability
        LIA(u)= outdeg(v)*weight(u,v)
    """
    lia = 0
    for path in paths:
        for i in range(len(path)):
            if i != len(path) - 1 :
                if len(path)==2:
                    lia = float(graph[path[i]][path[i + 1]]['weight'])
    return lia
def calculate_SIF(graph):
    for node1 in graph.nodes():
        reachable_nodes = graph.nodes[node1]['reachable_nodes']
        D = graph.nodes[node1]['D']
        L = graph.nodes[node1]['L']
        _sum = 0
        for r in reachable_nodes:
            _sum += graph.nodes[node1]['L']*graph.nodes[r]['L'] / (D[r] * D[r])
        graph.nodes[node1]['SIF'] = _sum
    return graph

def parse_graph(G):
    roots = list()
    roots_noleaf = list()
    str_list = list()
    probs = list()
    probs_noleaf = list()
    sif_sum_noleaf = 0.0
    sif_sum = 0.0
    for node in G.nodes():
        sif=G.nodes[node]['SIF']
        org_sif = sif
        if sif == 0: sif += pseudo_count
        sif_sum += sif
        if org_sif > 0:
            sif_sum_noleaf += sif
    for node in G.nodes():
        sif=G.nodes[node]['SIF']
        org_sif = sif
        if sif == 0: sif += pseudo_count
        roots.append(node)
        prob = sif / sif_sum
        probs.append(prob)
        if org_sif > 0:
            roots_noleaf.append(node)
            prob = sif / sif_sum_noleaf
            probs_noleaf.append(prob)
    sample_total = walks_per_graph
    first_time = True
    G = node2vec.Graph(G, True, p, q)
    G.preprocess_transition_probs()
    while True:
        if first_time:
            first_time = False
            node_list = roots
            prob_list = probs
        else:
            node_list = roots_noleaf
            prob_list = probs_noleaf
        n_sample = min(len(node_list), sample_total)
        if n_sample <= 0: break
        sample_total -= n_sample
        sampled_nodes = np.random.choice(node_list, n_sample, replace=False, p=prob_list)
        walks = G.simulate_walks(len(sampled_nodes), walk_length, sampled_nodes)
        for walk in walks:
            str_list.append(' '.join(str(k) for k in walk))

    return '\t'.join(str_list)

def writewalk(string,path):
    write_file = open(path, 'a')
    write_file.write(string + "\n")
    write_file.close()

def learn_embeddings(walks, embeding_vector_size):
  embed_file = 'node_vec_'+ str(embeding_vector_size) + ".txt"
  model = Word2Vec(walks, vector_size=embeding_vector_size, window=window_vector_size, min_count=0, sg=1, workers=worker,
                   epochs=iter)
  model.wv.save_word2vec_format(embed_file)

if __name__ == "__main__":

    count=1
    walks=set()
    string2=''
    start = time.time()
    for i0 in range(1,6):
        for i1 in range(i0,i0+7):
            g = Graph('1-'+str(i1)+'.txt')
            g = calculate_SIF(g)
            string=parse_graph(g)
            walks.add(string)
            writewalk(str(count) + '	' + str(string), 'random_walks_train.txt')
            count=count+1
    learn_embeddings(walks, dimensions)
    for i1 in range(6,13):
        g = Graph('1-'+str(i1)+'.txt')
        g = calculate_SIF(g)
        string=parse_graph(g)
        writewalk(str(count) + '	' + str(string), 'random_walks_val.txt')
        count = count + 1

    for i1 in range(7,14):
        g = Graph('1-' + str(i1) + '.txt')
        g = calculate_SIF(g)
        string = parse_graph(g)
        writewalk(str(count) + '	' + str(string), 'random_walks_test.txt')
        count = count + 1
    print("Finished!\n----------------------------------------------------------------")
    print("Time:", time.time() - start)
