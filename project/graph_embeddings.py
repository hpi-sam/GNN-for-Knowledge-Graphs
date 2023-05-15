import copy
import math
import pickle
import random

import networkx as nx
import torch
import torch_geometric.utils
from matplotlib import pyplot as plt
from numpy import dot
from numpy.linalg import norm
from scipy import stats
from torch_geometric.nn import GraphSAGE
from torchmetrics.functional import retrieval_normalized_dcg


def get_embedding(graph: torch_geometric.data.Data, model, query=('', '')):
    graph_sage = model

    edge_index = graph.edge_index
    embedding = graph_sage.forward(graph.x, edge_index, edge_weight=graph.edge_weight)

    node_embeddings = {node: node_embedding.data for node, node_embedding in zip(graph.y, embedding)}
    graph_embedding_super = node_embeddings['super_node']
    graph_embedding_average = (sum(node_embeddings.values()) - node_embeddings['super_node']) / (
            len(node_embeddings) - 1)

    query_embedding = (node_embeddings[query[0]], node_embeddings[query[1]]) if query[0] and query[1] else None

    return graph_embedding_super, graph_embedding_average, query_embedding


def convert_graph(graph: nx.DiGraph, keep_node_data=True):
    graph = graph.copy()

    for index, node in enumerate(graph.nodes(data=True)):
        node = node[-1]
        if keep_node_data:
            features = torch.FloatTensor(list(node.values()))
        else:
            features = torch.FloatTensor([1] + list(node.values())[-1:])
        node.clear()
        node['x'] = features

    if keep_node_data:
        graph.add_node('super_node', x=torch.FloatTensor([0, len(graph.nodes), 1, 0.5]))
    else:
        graph.add_node('super_node', x=torch.FloatTensor([1.0, 0.5]))

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


def stik_list_to_graph(graph: list, identifiers, query, predicate, visualize=False, title="") -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    existing_sources = []
    existing_targets = []

    for service in query:
        nx_graph.add_node(service['service'], starting_time=service['starting_time'],
                          ending_time=service['ending_time'], resource_consumption=service['resource_consumption'],
                          identifier=identifiers[service['service']])
        existing_sources.append(service['service'])
    for service in predicate:
        nx_graph.add_node(service['service'], starting_time=service['starting_time'],
                          ending_time=service['ending_time'], resource_consumption=service['resource_consumption'],
                          identifier=identifiers[service['service']])
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


def get_position_without_embedding(stik, source_node, target_node):
    sum_of_outgoing = 0
    sum_of_incoming = 0
    for source in stik:
        if source:
            sum_of_outgoing += sum([edge['probability'] for edge in source if
                                    edge['source_node'] == source_node and edge['target_node'] == target_node])
            sum_of_incoming += sum([edge['probability'] for edge in source if edge['target_node'] == target_node])

    if sum_of_incoming == 0:
        return 0

    return sum_of_outgoing / sum_of_incoming


def jaccard_similarity(data, ranking):
    data_set = set((index, element['rank']) for index, element in enumerate(data))
    ranking_set = set((index, element['rank']) for index, element in enumerate(ranking))
    intersection = len(ranking_set.intersection(data_set))
    union = len(ranking_set) + len(data_set) - intersection
    return float(intersection / union)


def cosine_similarity(data, ranking):
    data_ranks = [element['rank'] for element in data]
    ranking_ranks = [element['rank'] for element in ranking]
    return dot(data_ranks, ranking_ranks) / (norm(data_ranks) * norm(ranking_ranks))


def normalized_discounted_cumulative_gain(data, ranking):
    rank_targets = torch.FloatTensor([element['rank'] for element in data])
    rank_predictions = torch.FloatTensor([element['rank'] for element in ranking])
    return retrieval_normalized_dcg(rank_predictions, rank_targets)


def kendalltau_b(data, ranking):
    x = [line['rank'] for line in data]
    adjusted_ranking = get_adjusted_ranking(copy.deepcopy(ranking))
    adjusted_ranking = sorted(adjusted_ranking, key=lambda e: e['position'], reverse=True)
    y = [line['rank'] for line in adjusted_ranking]
    res = stats.kendalltau(x, y)
    return abs(res.statistic)


def generate_random_node_identifiers(nodes):
    random_node_identifiers = {node['service']: random.random() for node in nodes}
    return random_node_identifiers


def get_adjusted_ranking(ranking, metric='position'):
    ranking[0]['rank'] = 0
    for index, line in enumerate(ranking[1:]):
        if line[metric] == ranking[index][metric]:
            line['rank'] = ranking[index]['rank']
        else:
            line['rank'] = ranking[index]['rank'] + 1
    return ranking


def main():
    # ground_truth = gnn.main()
    # with open('data/ground_truth.pickle', 'wb') as handle:
    #     pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/ground_truth.pickle', 'rb') as handle:
        ground_truth = pickle.load(handle)

    in_channels = -1
    hidden_channels = 2
    num_layers = 4
    out_channels = 1

    # torch.save(graph_sage_model, 'data/model.pth')
    # graph_sage_model = torch.load('data/model.pth')

    SOURCE_NODE = 'A0'
    TARGET_NODE = 'B0'

    NUMBER_OF_MODELS = 100
    VISUALIZE = False  # Enabling visualization will make first ranking very costly but print all the STIKs

    random_node_identifiers = generate_random_node_identifiers(ground_truth['query'] + ground_truth['predicate'])

    super_node = {'jaccard': [], 'cosine': [], 'ndcg': [], 'kendalltau': []}
    average = {'jaccard': [], 'cosine': [], 'ndcg': [], 'kendalltau': []}
    timeless = {'jaccard': [], 'cosine': [], 'ndcg': [], 'kendalltau': []}
    query = {'jaccard': [], 'cosine': [], 'ndcg': [], 'kendalltau': []}

    for i in range(NUMBER_OF_MODELS):

        if i > 0 and VISUALIZE:
            VISUALIZE = False

        print(f'Ranking #{i + 1}/{NUMBER_OF_MODELS}')
        graph_sage_model = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels=out_channels)
        graph_sage_model_timeless = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels=out_channels)
        data = []
        for index, stik in enumerate(ground_truth[(SOURCE_NODE, TARGET_NODE)]):
            position = get_position_without_embedding(stik, SOURCE_NODE, TARGET_NODE)

            stik_nx = stik_list_to_graph(stik, random_node_identifiers, ground_truth['query'],
                                         ground_truth['predicate'],
                                         visualize=VISUALIZE, title=f'{index}')
            pyg_graph = convert_graph(stik_nx)
            pyg_graph_timeless = convert_graph(stik_nx, keep_node_data=False)

            embedding, embedding_average, embedding_query = get_embedding(pyg_graph, graph_sage_model,
                                                                          (SOURCE_NODE, TARGET_NODE))
            embedding_timeless, _, _ = get_embedding(pyg_graph_timeless, graph_sage_model_timeless,
                                                     (SOURCE_NODE, TARGET_NODE))

            graph_data = {'rank': index, 'position': position, 'embedding': embedding,
                          'embedding_average': embedding_average,
                          'embedding_timeless': embedding_timeless,
                          'embedding_query': embedding_query}
            data.append(graph_data)
            # print(f'Graph {index} done')

        adjusted_ranking = get_adjusted_ranking(copy.deepcopy(data))

        random_order = data.copy()
        random.shuffle(random_order)

        data_by_embedding = sorted(random_order, key=lambda x: abs(x['embedding'] - data[0]['embedding']))
        data_by_embedding_timeless = sorted(random_order,
                                            key=lambda x: abs(x['embedding_timeless'] - data[0]['embedding_timeless']))
        data_by_embedding_average = sorted(random_order,
                                           key=lambda x: abs(x['embedding_average'] - data[0]['embedding_average']))
        data_by_embedding_query = sorted(random_order,
                                         key=lambda x: abs(math.dist(x['embedding_query'], data[0]['embedding_query'])))

        super_node['jaccard'].append(jaccard_similarity(data, data_by_embedding))
        super_node['cosine'].append(cosine_similarity(data, data_by_embedding))
        super_node['ndcg'].append(normalized_discounted_cumulative_gain(data, data_by_embedding).item())
        super_node['kendalltau'].append(kendalltau_b(adjusted_ranking, data_by_embedding))

        average['jaccard'].append(jaccard_similarity(data, data_by_embedding_average))
        average['cosine'].append(cosine_similarity(data, data_by_embedding_average))
        average['ndcg'].append(normalized_discounted_cumulative_gain(data, data_by_embedding_average))
        average['kendalltau'].append(kendalltau_b(adjusted_ranking, data_by_embedding_average))

        timeless['jaccard'].append(jaccard_similarity(data, data_by_embedding_timeless))
        timeless['cosine'].append(cosine_similarity(data, data_by_embedding_timeless))
        timeless['ndcg'].append(normalized_discounted_cumulative_gain(data, data_by_embedding_timeless))
        timeless['kendalltau'].append(kendalltau_b(adjusted_ranking, data_by_embedding_timeless))

        query['jaccard'].append(jaccard_similarity(data, data_by_embedding_query))
        query['cosine'].append(cosine_similarity(data, data_by_embedding_query))
        query['ndcg'].append(normalized_discounted_cumulative_gain(data, data_by_embedding_query))
        query['kendalltau'].append(kendalltau_b(adjusted_ranking, data_by_embedding_query))

    # Random identifier hurt
    # Identifier by order hurt
    with open('data/metrics/super_node.pickle', 'wb') as handle:
        pickle.dump(super_node, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/metrics/average.pickle', 'wb') as handle:
        pickle.dump(average, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/metrics/timeless.pickle', 'wb') as handle:
        pickle.dump(timeless, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/metrics/query.pickle', 'wb') as handle:
        pickle.dump(query, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pass


if __name__ == '__main__':
    main()
