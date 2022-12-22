import xml.etree.ElementTree as ElementTree

import matplotlib.pyplot
import networkx
import yaml


def read_architecture_from_xml(filename) -> networkx.Graph:
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    network_graph = networkx.DiGraph()
    existing_nodes = set()
    for child in root:
        vertex_id = child.attrib["id"]

        if vertex_id not in existing_nodes:
            existing_nodes.add(vertex_id)
            network_graph.add_node(vertex_id)

        if len(child) > 0:
            for adjacent_vertex in child:
                adjacent_vertex_id = adjacent_vertex.attrib["vertex"]

                if adjacent_vertex_id not in existing_nodes:
                    existing_nodes.add(adjacent_vertex_id)
                    network_graph.add_node(adjacent_vertex_id)

                network_graph.add_edge(vertex_id, adjacent_vertex_id)

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
                        network_graph.add_node(host_name)
                if document.get("kind") == 'Service':
                    service_name = document.get("metadata").get("name")
                    host_name = document.get("spec").get("selector").get("app.kubernetes.io/name")
                    if host_name not in existing_hosts:
                        existing_hosts.add(host_name)
                        network_graph.add_node(host_name)
                    network_graph.add_edge(service_name, host_name)
                    network_graph.add_edge(host_name, service_name)
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


def get_knowledge_graph(architecture_diagram_file_name: str, deployment_description_file_name: str) -> networkx.Graph:
    knowledge_graph = read_architecture_from_xml("data/Architecture-Diagram.xml")
    apply_deployment(knowledge_graph, 'data/deployment.yaml')
    return knowledge_graph


def plot_graph(plottable_graph: networkx.Graph):
    color_map = ['red' if "worker" in node else "teal" for node in plottable_graph]
    networkx.layout.spring_layout(plottable_graph)
    networkx.draw_networkx(plottable_graph, node_color=color_map)
    matplotlib.pyplot.show()


if __name__ == '__main__':
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

    networkx.write_graphml_lxml(graph, 'data/Sock-shop.graphml')
