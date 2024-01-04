# architecture analysis
import numpy as np
import networkx as nx


def prune_architecture(genome, config):
    """
    Disables connections which don't connect to outputs.
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
    Returns:
        genome: the input genome with unused connections disabled.
    """
    # Define nodes and edges
    edges = []  # (source, target)
    for cg in genome.connections.values():
        if cg.enabled:
            source, target = cg.key
            edges.append((source, target))
    nodes = np.unique(edges)

    # Create graph
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Determine which nodes lie on paths to outputs
    all_paths = []
    for i in np.setdiff1d(nodes, config.genome_config.output_keys):
        for j in config.genome_config.output_keys:
            try:
                if nx.has_path(G, i, j):
                    paths = nx.all_simple_paths(G, source=i, target=j)
                    all_paths.extend(paths)
            except:
                pass
    valid_nodes = np.unique(sum(all_paths, []))

    # Disable unused connections
    invalid_nodes = np.setdiff1d(nodes, valid_nodes)
    invalid_connections = []
    for invalid_node in invalid_nodes:
        for cg in genome.connections.keys():
            if invalid_node in cg:
                invalid_connections.append(cg)

    for cg in invalid_connections:
        genome.connections[cg].enabled = False

    return genome


def architecture_metrics(genome, config, channels):
    """
    Describes an architecture via several metrics.
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
        channels:
    Returns:
        metrics: a dictionary of metrics.
    Metrics:
        nodes_n: # of nodes,
        nodes_input: # of input nodes,
        nodes_hidden: # of hidden nodes,
        nodes_output: # of output nodes,
        edges_n: # of connections/edges,
        edges_f: # of forward edges,
        edges_r: # of recurrent (self-self) edges,
        edges_l: # of lateral (within group) edges,
        edges_b: # of backwards edges,
        edges_e_i: ratio of positive:negative weights,
        multi_uni_r: ratio of multimodal:unimodal units (hidden + output)
    """
    # Define data
    edges = []  # (source, target, weight)
    for cg in genome.connections.values():
        if cg.enabled:
            source, target = cg.key
            edges.append((source, target, cg.weight))
    edges = np.array(edges)
    nodes = np.unique(edges[:, :2])

    # Node features
    nodes_n = len(nodes)  # number of nodes
    nodes_input = sum(nodes < 0)  # number of input nodes
    nodes_hidden = sum(
        nodes > max(config.genome_config.output_keys)
    )  # number of hidden nodes
    nodes_output = sum(
        (nodes >= 0) & (nodes <= max(config.genome_config.output_keys))
    )  # number of output nodes
    assert nodes_input + nodes_hidden + nodes_output == nodes_n, "Misclassified nodes"

    # Edge features
    edges_n = edges.shape[0]  # number of edges
    edges_e_i = (edges[:, -1] > 0).sum() / edges_n  # ratio of positive:negative edges

    # Edge directions
    node_types = np.ones_like(edges[:, :2]) * 2.0  # output nodes
    node_types[
        edges[:, :2] > max(config.genome_config.output_keys)
    ] = 1.0  # hidden nodes
    node_types[edges[:, :2] < 0] = 0.0  # input nodes

    edge_directions = []
    for e, edge in enumerate(node_types):
        if edge[0] < edge[1]:
            edge_directions.append("F")  # forwards

        elif edge[0] == edge[1]:
            if edges[e, 0] == edges[e, 1]:
                edge_directions.append("R")  # recurrent
            else:
                edge_directions.append("L")  # lateral

        elif edge[0] > edge[1]:
            edge_directions.append("B")  # backwards
    assert len(edge_directions) == edges.shape[0], "Misclassified edges"

    edges_f = edge_directions.count("F")
    edges_r = edge_directions.count("R")
    edges_l = edge_directions.count("L")
    edges_b = edge_directions.count("B")

    # Multimodal features
    n_channels = sum(channels)  # input feature

    # Define graph
    G = nx.DiGraph()
    G.add_edges_from(list(edges[:, :2].astype(int)))

    # Define variables
    input_channels = np.ones(len(config.genome_config.input_keys), dtype=int)
    input_channels[::n_channels] = 0
    non_input_nodes = nodes[nodes >= 0]
    node_channels = np.zeros((n_channels, len(non_input_nodes)), dtype=int)

    for a, i in enumerate(config.genome_config.input_keys):
        for b, j in enumerate(non_input_nodes.astype(int)):
            try:
                if nx.has_path(G, i, j):
                    node_channels[input_channels[a], b] = 1.0
            except:
                pass
    multimodal_nodes = np.sum(node_channels, axis=0) > 1
    multi_uni_r = sum(multimodal_nodes) / len(multimodal_nodes)

    # Output
    metrics = {
        "nodes_n": nodes_n,
        "nodes_input": nodes_input,
        "nodes_hidden": nodes_hidden,
        "nodes_output": nodes_output,
        "edges_n": edges_n,
        "edges_f": edges_f,
        "edges_r": edges_r,
        "edges_l": edges_l,
        "edges_b": edges_b,
        "edges_e_i": edges_e_i,
        "multi_uni_r": multi_uni_r,
    }

    return metrics
