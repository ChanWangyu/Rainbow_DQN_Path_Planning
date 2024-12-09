
import numpy as np


import karateclub as kc

import networkx as nx
import time


if __name__ == "__main__":
    # G = nx.read_edgelist(r'../data/manhattan/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    # G = nx.read_edgelist(r'../../data/manhattan/partial_edges_index.txt',
    #                      create_using=nx.DiGraph(), nodetype=int, data=[('weight', float)])
    G = nx.read_graphml(r'../../data/random_graph.graphml', node_type=int, force_multigraph=True)
    print(G)
    # numeric_indices = [index for index in range(G.number_of_nodes())]
    # node_indices = sorted([node for node in G.nodes()])
    # print(numeric_indices, node_indices)
    model = kc.Node2Vec(dimensions=32)
    start = time.time()
    model.fit(G)
    end = time.time()
    print(model.get_embedding(), model.get_embedding().shape, end - start)
    np.save(r'./embed_data/random_node_embed.npy', model.get_embedding())
    # print(embeddings)
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
