import copy
import functools
from itertools import permutations, combinations

import networkx as nx
import numpy as np

import graph_embeddings
import interference_subgraph_generation
import knowledge_graph_utils


def main():
    knowledge_graph = knowledge_graph_utils.get_knowledge_graph('data/Architecture-Diagram_Interference.xml',
                                                                'data/deployment_Interference.yaml')

    architecture = knowledge_graph_utils.get_architecture_callgraph(knowledge_graph)
    deployment = knowledge_graph_utils.get_deployment_graph(knowledge_graph)

    static_interference_paths = interference_subgraph_generation.get_all_service_paths(knowledge_graph)
    # temporal_paths = create_temporal_callgraph(architecture)
    temporal_paths = np.load('data/temporal_callgraph.npy', allow_pickle=True)

    temporal_node_mapping = generate_temporal_node_mapping(static_interference_paths, temporal_paths)
    spatio_temporal_graphs = generate_spatio_temporal_graphs(knowledge_graph, temporal_node_mapping)
    spatio_temporal_graphs_interference_only = generate_spatio_temporal_graphs(knowledge_graph, temporal_node_mapping,
                                                                               interference_edge_only=True)

    distinct_paths = interference_subgraph_generation.get_distinct_paths(
        knowledge_graph_utils.get_architecture_callgraph(knowledge_graph))
    colors = ['teal', 'purple', 'green']
    path_map = {}
    for index, path in enumerate(distinct_paths):
        assert len(colors) <= len(distinct_paths)
        for node in path:
            path_map[node] = colors[index]
    color_map = ['red' if "worker" in node else path_map[node] for node in knowledge_graph]

    knowledge_graph_utils.plot_graph(knowledge_graph, layout=nx.layout.kamada_kawai_layout, color_map=color_map)
    knowledge_graph_utils.plot_graph(architecture, layout=nx.layout.spring_layout)
    knowledge_graph_utils.plot_graph(deployment, layout=nx.layout.planar_layout)

    example_graph = spatio_temporal_graphs[0]
    nx.write_graphml(example_graph, 'data/spatio_temporal_graph_example.graphml', named_key_ids=True)

    temporal_labels = {node: f'{time}:{node}' for node, time in nx.get_node_attributes(example_graph, 'time').items()}
    relabeled_graph = nx.relabel_nodes(example_graph, temporal_labels)
    knowledge_graph_utils.plot_graph(relabeled_graph, layout=nx.layout.kamada_kawai_layout)

    example_graph_interference_only = spatio_temporal_graphs_interference_only[0]
    temporal_labels_interference_only = {node: f'{time}:{node}' for node, time in
                                         nx.get_node_attributes(example_graph_interference_only, 'time').items()}
    relabeled_graph_interference_only = nx.relabel_nodes(example_graph_interference_only,
                                                         temporal_labels_interference_only)
    knowledge_graph_utils.plot_graph(relabeled_graph_interference_only, layout=nx.layout.kamada_kawai_layout)

    pass


def generate_temporal_node_mapping(static_interference_graphs, temporal_call_graphs):
    temporal_node_mapping = set()
    for static_graph in static_interference_graphs:
        services = tuple(filter(lambda node: 'worker' not in node, static_graph))
        worker = [(index, node) for index, node in enumerate(static_graph) if 'worker' in node][0]
        valid_paths = tuple(filter(lambda possible_path: set(services).issubset(possible_path), temporal_call_graphs))
        for path in valid_paths:
            path = list(path)
            node_after_worker = static_graph[worker[0] + 1]
            node_before_worker = static_graph[worker[0] - 1]
            if path.index(node_before_worker) < path.index(node_after_worker):
                path.insert(path.index(node_after_worker), worker[1])
                temporal_node_map = tuple(
                    (node, time, node == node_before_worker or node == node_after_worker) for time, node in
                    enumerate(path))
                temporal_node_mapping.add(temporal_node_map)

    with open('data/mappings.npy', 'wb') as outfile:
        np.save(outfile, np.array(list(temporal_node_mapping), dtype=object))
    return temporal_node_mapping


def generate_spatio_temporal_graphs(knowledge_graph: nx.DiGraph, temporal_node_mapping,
                                    interference_edge_only: bool = False) -> list[nx.DiGraph]:
    spatio_temporal_graphs = []
    for temporal_node_map in temporal_node_mapping:
        temporal_node_dict = {node: time for node, time, _ in temporal_node_map}
        if interference_edge_only:
            interference_node_dict = {node: interference for node, _, interference in temporal_node_map}
        spatio_temporal_graph = nx.DiGraph(knowledge_graph)
        for node in knowledge_graph.nodes:
            if node in temporal_node_dict:
                spatio_temporal_graph.nodes[node]['time'] = temporal_node_dict[node]
            else:
                spatio_temporal_graph.remove_node(node)

        if interference_edge_only:
            for edge in nx.DiGraph(spatio_temporal_graph).edges:
                if 'worker' in edge[0]:
                    if edge[1] not in interference_node_dict or not interference_node_dict[edge[1]]:
                        spatio_temporal_graph.remove_edge(*edge)
                elif 'worker' in edge[1]:
                    if edge[0] not in interference_node_dict or not interference_node_dict[edge[0]]:
                        spatio_temporal_graph.remove_edge(*edge)

        spatio_temporal_graphs.append(spatio_temporal_graph)

    return spatio_temporal_graphs


def create_temporal_callgraph(call_graph: nx.DiGraph, file_suffix: str = ""):
    paths = interference_subgraph_generation.get_distinct_paths(call_graph)
    temporal_paths = []
    for combination in combinations(paths, 2):
        temporal_paths.extend(permute_paths(*combination))
    with open(f'data/temporal_callgraph{file_suffix}.npy', 'wb') as outfile:
        np.save(outfile, np.array(temporal_paths, dtype=object))
    return temporal_paths


def permute_paths(a, b):
    '''Find all possible permutations for order of calls in two paths'''
    concatenated = a + b
    result = []
    for permutation in permutations(concatenated):
        valid_permutation = True
        iter_a = 0
        iter_b = 0
        for element in permutation:
            if element in a:
                if a.index(element) < iter_a:
                    valid_permutation = False
                    break
                iter_a = a.index(element)
            if element in b:
                if b.index(element) < iter_b:
                    valid_permutation = False
                    break
                iter_b = b.index(element)
        if valid_permutation:
            result.append(permutation)
    return result


def embedding():
    knowledge_graph = knowledge_graph_utils.get_knowledge_graph('data/Architecture-Diagram_Interference.xml',
                                                                'data/deployment_Interference.yaml')

    graph_embeddings.get_embedding(knowledge_graph)

    knowledge_graph_utils.plot_graph(knowledge_graph, layout=nx.layout.kamada_kawai_layout)


def pseudo_code_implementation():
    knowledge_graph = knowledge_graph_utils.get_knowledge_graph(
        'data/Architecture-Diagram_Interference_Simple_Nodes.xml',
        'data/deployment_Interference_Simple_Nodes.yaml')

    architecture = knowledge_graph_utils.get_architecture_callgraph(knowledge_graph)
    deployment = knowledge_graph_utils.get_deployment_graph(knowledge_graph)

    distinct_paths = interference_subgraph_generation.get_distinct_paths(
        knowledge_graph_utils.get_architecture_callgraph(knowledge_graph))
    colors = ['teal', 'purple', 'green']
    path_map = {}
    for index, path in enumerate(distinct_paths):
        assert len(colors) <= len(distinct_paths)
        for node in path:
            path_map[node] = colors[index]
    color_map = ['red' if "worker" in node else path_map[node] for node in knowledge_graph]

    knowledge_graph_utils.plot_graph(knowledge_graph, layout=nx.layout.kamada_kawai_layout, color_map=color_map)
    knowledge_graph_utils.plot_graph(architecture, layout=nx.layout.spring_layout)
    knowledge_graph_utils.plot_graph(deployment, layout=nx.layout.planar_layout)

    query_predicate_stacks = generate_query_predicate_stacks(knowledge_graph, 'worker1')

    query_predicate_impact_lists = []

    for query_predicate_pair in query_predicate_stacks:
        query_stack = query_predicate_pair['query']
        predicate_stack = query_predicate_pair['predicate']

        query_predicate_impact_list = compute_list_of_impacted_pairs(query_stack, predicate_stack)
        query_predicate_impact_lists.append(query_predicate_impact_list)

    ground_truth = generate_ground_truth(query_predicate_impact_lists, query_predicate_stacks)
    ground_truth['query'] = query_stack
    ground_truth['predicate'] = predicate_stack

    return ground_truth


def generate_query_predicate_stacks(knowledge_graph, host_node):
    architecture = knowledge_graph_utils.get_architecture_callgraph(knowledge_graph)
    deployment = knowledge_graph_utils.get_deployment_graph(knowledge_graph)

    # execution_orders = create_temporal_callgraph(architecture, file_suffix='_simple_nodes')
    execution_orders = np.load('data/temporal_callgraph_simple_nodes.npy', allow_pickle=True)
    service_paths = interference_subgraph_generation.get_distinct_paths(architecture)
    path_of_service = {service: index for index, path in enumerate(service_paths) for service in path}

    nodes_at_host = [node for node in deployment.nodes if [node, host_node] in deployment.edges]
    execution_orders_at_host = list(
        filter(lambda execution_order: all(node in execution_order for node in nodes_at_host), execution_orders))

    query_predicate_stacks = []
    for execution_order in execution_orders_at_host:
        path_a_index = path_of_service[execution_order[0]]
        query = []
        predicate = []
        for time, service in enumerate(execution_order):
            stack_entry = {'service': service, 'starting_time': 0, 'ending_time': time + 1, 'resource_consumption': 1}
            if path_of_service[service] == path_a_index:
                query.append(stack_entry)
            else:
                predicate.append(stack_entry)

        for index, entry in enumerate(query[1:]):
            entry['starting_time'] = query[index]['ending_time']

        for index, entry in enumerate(predicate[1:]):
            entry['starting_time'] = predicate[index]['ending_time']

        query_predicate_stacks.append({'query': query, 'predicate': predicate})

    return query_predicate_stacks


def compute_list_of_impacted_pairs(source_stack, target_stack):
    sorted_source_stack = sorted(copy.deepcopy(source_stack), key=lambda entry: entry['starting_time'], reverse=True)
    sorted_target_stack = sorted(copy.deepcopy(target_stack), key=lambda entry: entry['starting_time'], reverse=True)
    result_list = []

    while sorted_source_stack:
        current_source_node = sorted_source_stack.pop()
        current_target_list = []
        while sorted_target_stack and current_source_node['ending_time'] > sorted_target_stack[-1]['starting_time']:
            current_target_node = sorted_target_stack.pop()
            current_target_list.append(current_target_node.copy())
            if current_source_node['ending_time'] < current_target_node['ending_time']:
                current_target_node['starting_time'] = current_source_node['ending_time']
                sorted_target_stack.append(current_target_node)
        result_list.append(compute_probability_and_magnitude_of_interference(current_source_node, current_target_list))

    return result_list


def compute_probability_and_magnitude_of_interference(source_node, target_list):
    total_source_time = source_node['ending_time'] - source_node['starting_time']
    result = []
    for current_target_node in target_list:
        total_target_time = min(current_target_node['ending_time'], source_node['ending_time']) - current_target_node[
            'starting_time']
        current_magnitude = source_node['resource_consumption'] + current_target_node['resource_consumption']
        if current_magnitude < 1:
            current_magnitude = 0
        else:
            current_magnitude = current_magnitude - 1
        result.append({'source_node': source_node['service'], 'target_node': current_target_node['service'],
                       'probability': total_target_time / total_source_time, 'magnitude': current_magnitude})
    return result


def expected_impact_of_query_stack(query_predicate_impact_list):
    result = {}
    for query_node in query_predicate_impact_list:
        result[query_node[0]['source_node']] = sum([entry['magnitude'] * entry['probability'] for entry in query_node])
    return result


def sum_of_probability(query_preducate_impact_list, target_node):
    sum_of_probabilities = 0
    for entry in query_preducate_impact_list:
        for element in entry:
            if element['target_node'] == target_node:
                sum_of_probabilities += element['probability']
    return sum_of_probabilities


def generate_ground_truth(query_predicate_impact_lists, query_predicate_stacks):
    query_services = [entry['service'] for entry in query_predicate_stacks[0]['query']]
    predicate_services = [entry['service'] for entry in query_predicate_stacks[0]['predicate']]
    source_target_combinations = [(a, b) for a in query_services for b in predicate_services]

    rankings = {}
    for combination in source_target_combinations:
        rankings[combination] = rank_graphs(*combination, query_predicate_impact_lists)

    return rankings


def rank_graphs(source, target, graphs: list):
    return sorted(graphs,
                  key=functools.cmp_to_key(lambda graphA, graphB: compare_graphs(graphA, graphB, source, target)))


def compare_graphs(a, b, source, target):
    relevant_entry_a = list(filter(len, a))
    relevant_entry_b = list(filter(len, b))
    relevant_entry_a = list(filter(lambda x: x[0]['source_node'] == source, relevant_entry_a))
    relevant_entry_b = list(filter(lambda x: x[0]['source_node'] == source, relevant_entry_b))

    if relevant_entry_a and relevant_entry_b:
        relevant_entry_a = relevant_entry_a[0]
        relevant_entry_b = relevant_entry_b[0]
        probability_a = sum([entry['probability'] for entry in relevant_entry_a if entry['target_node'] == target])
        probability_b = sum([entry['probability'] for entry in relevant_entry_b if entry['target_node'] == target])
        if probability_b - probability_a == 0:
            impact_a = sum_of_probability(a, target)
            impact_b = sum_of_probability(b, target)
            return impact_a - impact_b

        else:
            return probability_b - probability_a
    else:
        if len(relevant_entry_a) == len(relevant_entry_b):
            return 0
        elif not relevant_entry_a:
            return 1
        else:
            return -1


if __name__ == '__main__':
    pseudo_code_implementation()
