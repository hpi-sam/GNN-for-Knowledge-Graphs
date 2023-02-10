import networkx as nx
import matplotlib.pyplot as plt

import knowledge_graph_utils


def main():
    knowledge_graph = knowledge_graph_utils.get_knowledge_graph('data/Architecture-Diagram_Interference.xml',
                                                                'data/deployment_Interference.yaml')
    knowledge_graph_utils.plot_graph(knowledge_graph, integer_labels=False, layout=nx.layout.kamada_kawai_layout)

    nx.draw_kamada_kawai(knowledge_graph)
    plt.show()


if __name__ == '__main__':
    main()
