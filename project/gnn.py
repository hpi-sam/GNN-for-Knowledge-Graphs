import interference_subgraph_generation
import knowledge_graph_utils


def main():
    knowledge_graph = knowledge_graph_utils.get_knowledge_graph('data/Architecture-Diagram_Interference.xml',
                                                                'data/deployment_Interference.yaml')

    architecture = knowledge_graph_utils.read_architecture_from_xml('data/Architecture-Diagram_Interference.xml')

    # knowledge_graph_utils.plot_graph(knowledge_graph, layout=nx.layout.kamada_kawai_layout)
    # knowledge_graph_utils.plot_graph(architecture, layout=nx.layout.spring_layout)

    distinct_paths = interference_subgraph_generation.get_distinct_paths(architecture)
    interference_subgraph_generation.get_all_service_paths(knowledge_graph, distinct_paths)


if __name__ == '__main__':
    main()
