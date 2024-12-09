
import numpy as np

from envs.graph_embed.tool.classify import read_node_label, Classifier
from envs.graph_embed import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import time


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # G = nx.read_edgelist(r'../data/manhattan/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    G = nx.read_edgelist(r'../../data/manhattan/manhattan_edges_list.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', float)])
    print(G)
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    start = time.time()
    model.train(window_size=5, iter=3)
    end = time.time()
    print('time: ', end - start)
    embeddings = model.get_embeddings()
    print(embeddings, embeddings['10783732092'].shape)
    # print(embeddings)
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
