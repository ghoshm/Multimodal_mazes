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


def define_layers(genome, config):
    """
    Assigns each hidden node to a layer.
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
    Returns:
        h_nodes: each hidden nodes key/label.
        layers: each hidden nodes layer.
                If there are no hidden nodes, it will return empty lists.
    Note this is conceptually similar to a topological sort.
    Though allows for directed cycles.
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

    # Define layers
    h_nodes, layers = [], []
    for i in nodes[
        nodes > max(config.genome_config.output_keys)
    ]:  # for all hidden units
        path_lengths = []
        for j in list(
            set(nodes) & set(config.genome_config.output_keys)
        ):  # for used output units
            if nx.has_path(G, i, j):
                paths = nx.all_simple_paths(G, source=i, target=j)
                path_lengths.extend([len(path) - 1 for path in paths])
        v, c = np.unique(path_lengths, return_counts=True)
        layers.append(v[np.argmax(c)])
        h_nodes.append(i)

    if layers:
        layers = (max(layers) - np.array(layers)) + 1

    return np.array(h_nodes), np.array(layers)


def architecture_metrics(genome, config, channels):
    """
    Describes an architecture via several metrics, which
    are grouped into two dictionaries (mostly for plotting purposes).
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
        channels:
    Returns:
        metrics_n: a dictionary of metrics with variable limits.
        metrics_p: a dictionary of metrics between 0 and 1.

    Metrics_n:
        Number of:
            nodes_n: nodes.
            nodes_input: input nodes.
            nodes_hidden: hidden nodes.
            nodes_output: output nodes.
            edges_n: connections/edges.
            layers_n: layers (inc inputs + outputs)
        in_degree: mean in degree (excluding input units).
        out_degree: mean out degree (excluding output units).

    Metrics_p:
        Proportion of:
            edges_f: forward edges.
            edges_fs: forward skip edges.
            edges_r: recurrent (self-self) edges.
            edges_l: lateral (within layer) edges.
            edges_b: backwards edges.
            edges_bs: backwards skip edges.
            density: connections present.
            reciprocity: reciprocal connections (a-b-a)
        transitivity: transitivity ().
        edges_e_i: ratio of positive:negative weights.
        multi_uni_r: ratio of multimodal:unimodal units (hidden + output).
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
    node_types = np.zeros_like(edges[:, :2])  # inputs
    h_nodes, layers = define_layers(genome, config)
    for a, h_node in enumerate(h_nodes):
        node_types[edges[:, :2] == h_node] = layers[a]  # hidden
    node_types[
        (edges[:, :2] >= 0) & (edges[:, :2] <= max(config.genome_config.output_keys))
    ] = (
        node_types.max() + 1
    )  # outputs

    edge_types = []
    for e, edge in enumerate(node_types):
        if edge[0] < edge[1]:
            if edge[0] - edge[1] == -1.0:
                edge_types.append("F")  # forwards
            else:
                edge_types.append("FS")  # forwards skip

        elif edge[0] == edge[1]:
            if edges[e, 0] == edges[e, 1]:
                edge_types.append("R")  # recurrent (self-self)
            else:
                edge_types.append("L")  # lateral

        elif edge[0] > edge[1]:
            if edge[0] - edge[1] == 1.0:
                edge_types.append("B")  # backwards
            else:
                edge_types.append("BS")  # backwards skip

    assert len(edge_types) == edges.shape[0], "Misclassified edges"

    edges_f = np.array(edge_types.count("F")) / len(edge_types)
    edges_fs = np.array(edge_types.count("FS")) / len(edge_types)
    edges_r = np.array(edge_types.count("R")) / len(edge_types)
    edges_l = np.array(edge_types.count("L")) / len(edge_types)
    edges_b = np.array(edge_types.count("B")) / len(edge_types)
    edges_bs = np.array(edge_types.count("BS")) / len(edge_types)

    # Multimodal features
    n_channels = sum(channels)  # input channels

    # Define graph
    G = nx.DiGraph()
    G.add_edges_from(list(edges[:, :2].astype(int)))

    # Define variables
    input_channels = np.ones(len(config.genome_config.input_keys), dtype=int)
    input_channels[::n_channels] = 0
    non_input_nodes = nodes[nodes >= 0].astype(int)
    node_channels = np.zeros((n_channels, len(non_input_nodes)), dtype=int)

    for a, i in enumerate(config.genome_config.input_keys):
        for b, j in enumerate(non_input_nodes):
            try:
                if nx.has_path(G, i, j):
                    node_channels[input_channels[a], b] = 1.0
            except:
                pass
    multimodal_nodes = np.sum(node_channels, axis=0) > 1
    multi_uni_r = sum(multimodal_nodes) / len(multimodal_nodes)

    # Graph metrics
    # Mean in degree
    tmp = []
    for node in nodes[nodes >= 0]:  # excluding input nodes
        tmp.append(G.in_degree(node))
    in_degree_mean = np.array(tmp).mean()

    # Mean out degree
    tmp = []
    for node in np.setdiff1d(nodes, config.genome_config.output_keys):
        tmp.append(G.out_degree(node))
    out_degree_mean = np.array(tmp).mean()

    # Density
    density = len(edges) / (
        (len(nodes[nodes < 0]) * len(nodes[nodes >= 0])) + (len(nodes[nodes >= 0]) ** 2)
    )  # between 0 (sparse) and 1 (dense)

    # Reciprocity
    reciprocity = nx.overall_reciprocity(
        G
    )  # between 0 (feedforward) and 1 (bidirectional)

    # Transitivity
    transitivity = nx.transitivity(G)  # between 0 and 1

    # Output
    metrics_n = {
        "nodes_n": nodes_n,
        "nodes_input": nodes_input,
        "nodes_hidden": nodes_hidden,
        "nodes_output": nodes_output,
        "edges_n": edges_n,
        "layers_n": max(np.append(layers, 0)) + 2,
        "in_degree": in_degree_mean,
        "out_degree": out_degree_mean,
    }

    metrics_p = {
        "edges_f": edges_f,
        "edges_fs": edges_fs,
        "edges_r": edges_r,
        "edges_l": edges_l,
        "edges_b": edges_b,
        "edges_bs": edges_bs,
        "density": density,
        "reciprocity": reciprocity,
        "transitivity": transitivity,
        "edges_e_i": edges_e_i,
        "multi_uni_r": multi_uni_r,
    }

    return metrics_n, metrics_p
