import networkx as nx

import interference_subgraph_generation
import knowledge_graph_utils


def main():
    knowledge_graph = knowledge_graph_utils.get_knowledge_graph('data/Architecture-Diagram_Interference.xml',
                                                                'data/deployment_Interference.yaml')

    static_interference_paths = interference_subgraph_generation.get_all_service_paths(knowledge_graph)

    architecture = knowledge_graph_utils.get_architecture_callgraph(knowledge_graph)
    deployment = knowledge_graph_utils.get_deployment_graph(knowledge_graph)

    distinct_paths = interference_subgraph_generation.get_distinct_paths(
        knowledge_graph_utils.get_architecture_callgraph(knowledge_graph))
    colors = ['teal', 'purple', 'green']
    path_map = {}
    for index, path in enumerate(distinct_paths):
        assert len(colors) <= len(distinct_paths)
        for node in path[0]:
            path_map[node] = colors[index]
    color_map = ['red' if "worker" in node else path_map[node] for node in knowledge_graph]

    knowledge_graph_utils.plot_graph(knowledge_graph, layout=nx.layout.kamada_kawai_layout, color_map=color_map)
    knowledge_graph_utils.plot_graph(architecture, layout=nx.layout.spring_layout)
    knowledge_graph_utils.plot_graph(deployment, layout=nx.layout.planar_layout)

    print(static_interference_paths)


    print(color_map)


if __name__ == '__main__':
    main()
