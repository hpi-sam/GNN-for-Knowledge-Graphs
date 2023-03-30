import pickle

import networkx as nx
import torch
import torch_geometric.utils
from matplotlib import pyplot as plt
from torch_geometric.nn import GraphSAGE


def get_embedding(graph: torch_geometric.data.Data):
    in_channels = -1
    hidden_channels = 1
    num_layers = 2
    out_channels = 1
    graph_sage = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels=out_channels)
    print(graph_sage)
    node_features = graph.num_node_features
    edge_index = graph.edge_index
    embedding = graph_sage.forward(graph.x, edge_index, edge_weight=graph.edge_weight)

    node_embeddings = {node: node_embedding.data for node, node_embedding in zip(graph.y, embedding)}
    pass


def convert_graph(graph: nx.DiGraph):
    for node in graph.nodes(data=True):
        node = node[-1]
        features = torch.FloatTensor(list(node.values()))
        node.clear()
        node['x'] = features

    graph.add_node('super_node', x=torch.FloatTensor([0, len(graph.nodes), 1]))
    for node in graph.nodes:
        if node != 'super_node':
            graph.add_edge(node, 'super_node', weight=1, type='super')

    for node in graph.nodes:
        graph.nodes[node]['y'] = node
    graph = nx.convert_node_labels_to_integers(graph)

    nodes = graph.nodes(data=True)
    edges = graph.edges(data=True)

    # for edge in graph.edges(data=True):
    #     edge = edge[-1]
    #     attributes = [1. if edge['type'] == 'interference' else 0.]
    #     edge.clear()
    #     edge['x'] = attributes

    # for _, _, d in graph.edges(data=True):
    #     d.clear()
    # torch_graph.edge_attr = torch.tensor([1 if edge_type == 'call' else 2 for edge_type in torch_graph.type])

    torch_graph = torch_geometric.utils.from_networkx(graph)
    torch_graph.edge_weight = torch.FloatTensor(torch_graph.weight)

    return torch_graph


def stik_list_to_graph(graph: list, query, predicate, visualize=False) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    existing_sources = []
    existing_targets = []

    for service in query:
        nx_graph.add_node(service['service'], starting_time=service['starting_time'],
                          ending_time=service['ending_time'], resource_consumption=service['resource_consumption'])
        existing_sources.append(service['service'])
    for service in predicate:
        nx_graph.add_node(service['service'], starting_time=service['starting_time'],
                          ending_time=service['ending_time'], resource_consumption=service['resource_consumption'])
        existing_targets.append(service['service'])

    for element in graph:
        if element:
            current_source = element[0]['source_node']
        for interference in element:
            current_target = interference['target_node']
            nx_graph.add_edge(current_source, current_target, weight=interference['probability'],
                              type='interference')

    existing_sources = sorted(existing_sources)
    existing_targets = sorted(existing_targets)
    pos = nx.bipartite_layout(nx_graph, existing_sources)
    pos_sorted = {}

    positions = []
    for node in existing_sources:
        positions.append(pos[node])
    positions = sorted(positions, key=lambda x: x[-1], reverse=True)
    for index, node in enumerate(existing_sources):
        pos_sorted[node] = positions[index]

    positions = []
    for node in existing_targets:
        positions.append(pos[node])
    positions = sorted(positions, key=lambda x: x[-1], reverse=True)
    for index, node in enumerate(existing_targets):
        pos_sorted[node] = positions[index]

    while len(existing_sources) > 1:
        current_node = existing_sources.pop()
        nx_graph.add_edge(existing_sources[-1], current_node, type='call', weight=1)
    while len(existing_targets) > 1:
        current_node = existing_targets.pop()
        nx_graph.add_edge(existing_targets[-1], current_node, type='call', weight=1)

    if visualize:
        plt.figure(3, figsize=(12, 12))
        nx.draw(nx_graph, pos=pos_sorted, with_labels=True, node_size=500)
        nx.draw_networkx_edge_labels(nx_graph, pos=pos_sorted, font_size=14)
        plt.show()

    return nx_graph


def main():
    # ground_truth = gnn.pseudo_code_implementation()
    # with open('data/ground_truth.pickle', 'wb') as handle:
    #     pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/ground_truth.pickle', 'rb') as handle:
        ground_truth = pickle.load(handle)

    stik_nx = stik_list_to_graph(ground_truth[('A0', 'B0')][0], ground_truth['query'], ground_truth['predicate'],
                                 visualize=True)
    pyg_graph = convert_graph(stik_nx)

    get_embedding(pyg_graph)


if __name__ == '__main__':
    main()
