import networkx
import numpy

import knowledge_graph_utils


def sample_subgraph(super_graph, start_node: str, bfs_prob=0.5, max_nodes=None, random_seed=42) -> networkx.DiGraph():
    """
    Static subgraph sampling algorithm (Taken from "Exploring Spatio-Temporal Graphs
    as Means to Identify Failure Propagation")

    @author Andrea Nathansen
    """
    numpy.random.seed(random_seed)
    subgraph = networkx.DiGraph()
    subgraph.add_node(start_node)
    nodes_to_visit = [start_node]
    if max_nodes is None:
        max_nodes = len(super_graph.nodes())
    assert max_nodes > 0, 'enter positive max_nodes'
    for node in nodes_to_visit:
        potential_next_nodes = [n for n in list(super_graph.neighbors(node)) if not n in nodes_to_visit]
        if len(potential_next_nodes) > 0:
            if numpy.random.choice([0, 1], p=[1 - bfs_prob, bfs_prob]):
                # bfs way
                next_nodes = potential_next_nodes
            else:
                # dfs way
                next_nodes = [potential_next_nodes[numpy.random.choice(len(potential_next_nodes))]]

            num_nodes_to_add = numpy.minimum(max_nodes - len(subgraph.nodes()), len(next_nodes))
            next_nodes = next_nodes[:num_nodes_to_add]
            nodes_to_visit += next_nodes
            for next_node in next_nodes:
                subgraph.add_node(next_node)
                subgraph.add_edge(node, next_node)
        if len(subgraph.nodes()) >= max_nodes:
            return subgraph
    return subgraph


def _has_received_propagation(subgraph, node):
    in_edges = list(subgraph.in_edges(node))
    all_timesteps = networkx.get_edge_attributes(subgraph, "timestep")
    in_edges_with_timesteps = [edge for edge in in_edges if edge in all_timesteps]
    return len(in_edges_with_timesteps) > 0


def _add_timesteps_to_graph(subgraph, start_node, random_seed=42):
    numpy.random.seed(random_seed)
    timestep = 0
    edge_pool = {edge: 1 for edge in list(subgraph.out_edges(start_node))}

    while edge_pool:
        edge_to_sample_idx = numpy.random.choice(range(len(edge_pool.keys())),
                                                 p=numpy.array(list(edge_pool.values())) / numpy.sum(
                                                     list(edge_pool.values())))
        edge_to_sample = list(edge_pool.keys())[edge_to_sample_idx]
        del edge_pool[edge_to_sample]
        networkx.set_edge_attributes(subgraph, {edge_to_sample: {"timestep": timestep}})
        for key in edge_pool.keys():
            edge_pool[key] += 1
        new_node = edge_to_sample[1]
        # don't propagate to the same node twice
        edge_pool.update({edge: 1 for edge in list(subgraph.out_edges(new_node)) if
                          not _has_received_propagation(subgraph, edge[1])})
        timestep += 1


def sample_temporal_subgraphs(super_graph, start_node, max_nodes=None, seed=numpy.random.seed(111), num_sub_seeds=1000):
    """
        Sampling Temporal Subgraphs (Taken from "Exploring Spatio-Temporal Graphs
        as Means to Identify Failure Propagation")

        @author Andrea Nathansen, lisakoeritz
        """
    if not max_nodes:
        max_nodes = len(super_graph.nodes()) + 1
    tn_subgraphs = []
    em = networkx.algorithms.isomorphism.numerical_edge_match("timestep", 0)
    for seed in numpy.random.choice(10000000, num_sub_seeds):
        for max_nodes_no in range(2, max_nodes):
            for bfs_prob in [0.1, 0.5, 0.9]:
                tn_subgraph = sample_subgraph(super_graph, start_node, max_nodes=max_nodes_no, bfs_prob=bfs_prob,
                                              random_seed=seed)
                _add_timesteps_to_graph(tn_subgraph, start_node, random_seed=seed)
                if tn_subgraphs:
                    if not numpy.any([networkx.is_isomorphic(tn_subgraph, s, edge_match=em) for s in tn_subgraphs]):
                        tn_subgraphs.append(tn_subgraph)
                else:
                    tn_subgraphs.append(tn_subgraph)
    return tn_subgraphs


if __name__ == '__main__':
    knowledge_graph = knowledge_graph_utils.get_knowledge_graph("data/Architecture-Diagram.xml", 'data/deployment.yaml')

    static_subgraph = sample_subgraph(knowledge_graph, "front-end", random_seed=12)
    knowledge_graph_utils.plot_graph(static_subgraph)

    # temporal_subgraphs1 = sample_temporal_subgraphs(static_subgraph, "front-end", num_sub_seeds=1)
    # temporal_subgraphs2 = sample_temporal_subgraphs(static_subgraph, "front-end", num_sub_seeds=2)
    temporal_subgraphs3 = sample_temporal_subgraphs(static_subgraph, "front-end", num_sub_seeds=3)

    last_subgraph = temporal_subgraphs3[len(temporal_subgraphs3) - 1]
    knowledge_graph_utils.plot_graph(last_subgraph)

    for edge in last_subgraph.edges:
        print(str(edge) + ": " + str(last_subgraph.get_edge_data(*edge)))

    networkx.write_graphml_lxml(last_subgraph, 'data/Sock-shop-temporal.graphml', named_key_ids=True)

    knowledge_graph_utils.plot_graph(temporal_subgraphs3)
    exit()
