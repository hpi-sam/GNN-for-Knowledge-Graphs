import random
import uuid

import numpy
import numpy as np

import knowledge_graph_utils
import subgraph_sampling

STARTING_SEED = 0


def main():
    knowledge_graph = knowledge_graph_utils.get_knowledge_graph("data/Architecture-Diagram.xml", 'data/deployment.yaml')
    services = [node for node in knowledge_graph if knowledge_graph.nodes[node].get('type') == 'service']
    random.seed = STARTING_SEED
    static_callgraphs = {node: [] for node in services}
    for node in static_callgraphs:
        start_node = node

        for i in range(10):
            static_callgraphs[node].append(
                subgraph_sampling.sample_subgraph(knowledge_graph, start_node, random_seed=STARTING_SEED + i))

    temporal_subgraphs = []
    for starting_node, graphs in static_callgraphs.items():
        print(starting_node, graphs)
        for graph in graphs:
            temporal_subgraphs.extend(
                subgraph_sampling.sample_temporal_subgraphs(graph, starting_node, num_sub_seeds=10))

    data = np.array(temporal_subgraphs, dtype=object)
    numpy.save('data/graphs_' + str(uuid.uuid1()), data)


if __name__ == '__main__':
    main()
