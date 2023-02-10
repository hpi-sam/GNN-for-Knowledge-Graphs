import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from node2vec import Node2Vec
from sklearn.manifold import TSNE

import knowledge_graph_utils
import subgraph_sampling


def node2vec_embedding(graph, dimensions=64, num_walks=10, workers=1):
    walk_length = len(graph.nodes)
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model.wv


def main():
    data = np.load('data/graphs_10x10.npy', allow_pickle=True)
    interference_subgraphs = tuple(filter(subgraph_sampling.detect_interference, data))

    print(len(data), len(interference_subgraphs))

    embedding = node2vec_embedding(interference_subgraphs[-1])
    nx.write_graphml_lxml(interference_subgraphs[-1], 'data/interference_subgraph.graphml', named_key_ids=True)
    knowledge_graph_utils.plot_graph(interference_subgraphs[-1], has_edge_labels=True)

    # print(embedding.most_similar('worker1'))

    print('embeddings:', len(embedding.vectors))

    # reduce dimensions to 2D for visualization
    embeddings_2d = TSNE(n_components=2, perplexity=10).fit_transform(embedding.vectors)

    # plot the node embeddings
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='tab:RED')
    plt.show()


if __name__ == '__main__':
    main()
