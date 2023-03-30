import copy
import xml.etree.ElementTree as ElementTree

import matplotlib.pyplot
import networkx
import yaml


def read_architecture_from_xml(filename) -> networkx.DiGraph:
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    network_graph = networkx.DiGraph()
    existing_nodes = set()
    for child in root:
        vertex_id = child.attrib["id"]

        if vertex_id not in existing_nodes:
            existing_nodes.add(vertex_id)
            network_graph.add_node(vertex_id, type='service')

        if len(list(child)) > 0:
            for adjacent_vertex in child:
                adjacent_vertex_id = adjacent_vertex.attrib["vertex"]

                if adjacent_vertex_id not in existing_nodes:
                    existing_nodes.add(adjacent_vertex_id)
                    network_graph.add_node(adjacent_vertex_id, type='service')

                network_graph.add_edge(vertex_id, adjacent_vertex_id, type="call")

    return network_graph


def apply_deployment(network_graph: networkx.Graph, filename: str):
    with open(filename, 'r') as deployment_file:
        deployment = yaml.load_all(deployment_file, yaml.Loader)
        existing_hosts = set()
        for document in deployment:
            if document:
                if document.get("kind") == 'Pod':
                    host_name = document.get("metadata").get("name")
                    if host_name not in existing_hosts:
                        existing_hosts.add(host_name)
                        network_graph.add_node(host_name, type='host')
                if document.get("kind") == 'Service':
                    service_name = document.get("metadata").get("name")
                    host_name = document.get("spec").get("selector").get("app.kubernetes.io/name")
                    if host_name not in existing_hosts:
                        existing_hosts.add(host_name)
                        network_graph.add_node(host_name)
                    network_graph.add_edge(service_name, host_name, type="host")
                    network_graph.add_edge(host_name, service_name, type="host")
    return network_graph


def move_service_to_host(network_graph: networkx.Graph, service: str, new_host: str):
    old_host = get_host(network_graph, service)
    network_graph.remove_edge(service, old_host)
    network_graph.remove_edge(old_host, service)
    network_graph.add_edge(service, new_host)
    network_graph.add_edge(new_host, service)


def switch_hosts(network_graph: networkx.Graph, service1: str, service2: str):
    old_host1 = get_host(network_graph, service1)
    old_host2 = get_host(network_graph, service2)
    move_service_to_host(network_graph, service1, old_host2)
    move_service_to_host(network_graph, service2, old_host1)


def get_host(network_graph: networkx.Graph, service: str) -> str:
    return [neighbour for neighbour in network_graph.neighbors(service) if "worker" in neighbour][0]


def get_knowledge_graph(architecture_diagram_file_name: str, deployment_description_file_name: str) -> networkx.DiGraph:
    knowledge_graph = read_architecture_from_xml(architecture_diagram_file_name)
    apply_deployment(knowledge_graph, deployment_description_file_name)
    return knowledge_graph


def get_architecture_callgraph(knowledge_graph: networkx.DiGraph) -> networkx.DiGraph:
    graph = copy.deepcopy(knowledge_graph)
    for edge in knowledge_graph.edges:
        if knowledge_graph.get_edge_data(*edge)['type'] == 'host':
            graph.remove_edge(*edge)
    for node in knowledge_graph.nodes:
        if graph.nodes[node]['type'] == 'host':
            graph.remove_node(node)
    return graph


def get_deployment_graph(knowledge_graph: networkx.DiGraph) -> networkx.DiGraph:
    graph = copy.deepcopy(knowledge_graph)
    for edge in knowledge_graph.edges:
        if knowledge_graph.get_edge_data(*edge)['type'] == 'call':
            graph.remove_edge(*edge)
    return graph


def plot_graph(plottable_graph: networkx.Graph, has_edge_labels: bool = False, integer_labels: bool = False,
               layout=networkx.layout.spring_layout, color_map=None, relabeled=False):
    if not color_map and not relabeled:
        color_map = ['red' if "worker" in node else "teal" for node in plottable_graph]
    if integer_labels:
        plottable_graph = networkx.convert_node_labels_to_integers(plottable_graph)
    # pos = layout(plottable_graph, k=3 ** (1 / 2))
    pos = layout(plottable_graph)
    networkx.draw_networkx(plottable_graph, pos, node_color=color_map)
    if has_edge_labels:
        edge_labels = {edge: plottable_graph.get_edge_data(*edge)["timestep"] for edge in plottable_graph.edges}
        networkx.draw_networkx_edge_labels(plottable_graph, pos, edge_labels=edge_labels)
    matplotlib.pyplot.show()


def main():
    graph = read_architecture_from_xml("data/Architecture-Diagram.xml")
    apply_deployment(graph, 'data/deployment.yaml')

    plot_graph(graph)

    print(get_host(graph, "user-db"))
    move_service_to_host(graph, "user-db", "worker1")
    print(get_host(graph, "user-db"))

    print("payment: " + get_host(graph, "payment"))
    print("session-db: " + get_host(graph, "session-db"))
    switch_hosts(graph, "payment", "session-db")
    print("payment: " + get_host(graph, "payment"))
    print("session-db: " + get_host(graph, "session-db"))

    print('Nodes', graph.nodes['orders'])

    networkx.write_graphml_lxml(graph, 'data/Sock-shop.graphml', named_key_ids=True)


if __name__ == '__main__':
    main()
