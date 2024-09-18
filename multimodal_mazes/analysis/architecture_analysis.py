# architecture analysis
import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
import multimodal_mazes


def prune_architecture(genome, config):
    """
    Disables connections which don't connect to outputs.
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
    Returns:
        genome: the input genome with unused connections disabled.
    """
    # Define graph
    G, nodes, edges = define_graph(genome, weights="None")

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
    # Define graph
    G, nodes, edges = define_graph(genome, weights="None")

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


def initial_architecture(initial_connection):
    """
    Create a DiGraph with an inital architecture,
        and uniform positive weights.
    Arguments:
        initial_connection: a string denoting the architecture to use.
    Returns:
        G: a nx.DiGraph with labelled nodes and weighted edges.
    """
    if initial_connection == "multimodal_half":
        edges = [[-1, 0], [-3, 1], [-2, 0], [-4, 1]]
    elif initial_connection == "multimodal":
        edges = [[-1, 0], [-3, 1], [-5, 2], [-7, 3], [-2, 0], [-4, 1], [-6, 2], [-8, 3]]

    nodes = np.unique(edges)

    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e[0], e[1], weight=1)

    for n in nodes:
        G.add_node(n, label=n)

    # list(G.edges(data=True)) # useful syntax for debugging

    return G


def edit_distance(genome, config):
    """
    Compute the edit distance between a network,
        and an inital architecture.
        Note: uses only the sign of the weights.
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
    Returns:
        edit_distance: the difference between the two.
    """
    # Define graph
    G, nodes, edges = define_graph(genome, weights="Binary")

    for n in nodes:
        G.add_node(n, label=n)

    # Edit distance
    em = iso.numerical_edge_match("weight", 1)
    nm = iso.numerical_node_match("label", 1)

    G_init = initial_architecture(config.genome_config.initial_connection)
    edit_distance = nx.graph_edit_distance(G, G_init, edge_match=em, node_match=nm)

    return edit_distance


def architecture_metrics(genome, config, channels):
    """
    Describes an architecture via several metrics, which
    are grouped into two dictionaries (mostly for plotting purposes).
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
        channels: list of active (1) and inative (0) channels e.g. [0,1].
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
        transitivity: if a-b, a-c, then b-c?
        square_clustering: if a-b, a-c, then b-d and c-d?
        edges_e_i: ratio of positive:negative weights.
        multi_uni_r: ratio of multimodal:unimodal units (hidden + output).
    """

    # Define graph
    G, nodes, edges = define_graph(genome, weights="None")

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

    # Mean square clustering
    square_clustering = nx.square_clustering(G)
    square_clustering_mean = np.mean(
        [square_clustering[n] for n in square_clustering.keys()]
    )

    # Output
    metrics_n = {
        "$\mathregular{\eta}$": nodes_n,
        "$\mathregular{\eta_{input}}$": nodes_input,
        "$\mathregular{\eta_{hidden}}$": nodes_hidden,
        "$\mathregular{\eta_{output}}$": nodes_output,
        "$\mathregular{E}$": edges_n,
        "$\mathregular{\iota}$": max(np.append(layers, 0)) + 2,
        "$\mathregular{\eta_{\overline{in}}}$": in_degree_mean,
        "$\mathregular{\eta_{\overline{out}}}$": out_degree_mean,
    }

    metrics_p = {
        "$\mathregular{i_\iota j_{\iota+1}}$": edges_f,
        "$\mathregular{i_\iota j_{\iota+n}}$": edges_fs,
        "$\mathregular{i i}$": edges_r,
        "$\mathregular{i_\iota j_\iota}$": edges_l,
        "$\mathregular{i_\iota j_{\iota-1}}$": edges_b,
        "$\mathregular{i_\iota j_{\iota-n}}$": edges_bs,
        "$\mathregular{D_{ensity}}$": density,
        "$\mathregular{R_{eciprocity}}$": reciprocity,
        "$\mathregular{C_{3}(G)}$": transitivity,
        "$\mathregular{C_{4}(G)}$": square_clustering_mean,
        "$\mathregular{E_{+}:E_{-}}$": edges_e_i,
        "$\mathregular{\eta_{multi}:\eta_{uni}}$": multi_uni_r,
    }

    return metrics_n, metrics_p


def architecture_metrics_matrices(agents, genomes, config):
    """
    Builds two matricies describing a list of agents.
        Note agents with <= 1 edge will be skipped.
    Arguments:
        agents: a list of indicies, of the genomes to test.
        genomes: neat generated genomes.
        config: the neat configuration holder.
    Returns:
        metrics_n: an agents x metrics np array with variable limits.
        metrics_p: an agents x metrics np array between 0 and 1.
        metrics_n.keys(): labels for metrics n.
        metrics_p.keys(): labels for metrics p.
    """

    agents_metrics_n, agents_metrics_p = [], []
    for g in agents:
        _, genome, channels = genomes[g]
        if genome.size()[1] > 1:
            genome = multimodal_mazes.prune_architecture(genome, config)

            if genome.size()[1] > 1:
                metrics_n, metrics_p = multimodal_mazes.architecture_metrics(
                    genome, config, channels
                )
                agents_metrics_n.append(list(metrics_n.values()))
                agents_metrics_p.append(list(metrics_p.values()))

    return (
        np.array(agents_metrics_n),
        np.array(agents_metrics_p),
        metrics_n.keys(),
        metrics_p.keys(),
    )


def define_graph(genome, weights):
    """
    Create a networkx graph from a genome,
        with either no or binary (-1/+1) weights.
    Arguments:
        genome: neat generated genome.
        weights: a string of:
            None: unweighted graph.
            Binary: weighted with (-1/+1).
    Returns:
        G: a networkx graph.
        nodes: a np vector of unique nodes.
        edges: a np array of edges n(source, target, weight).
    """

    # Define nodes and edges
    edges = []  # (source, target, weight)
    for cg in genome.connections.values():
        if cg.enabled:
            source, target = cg.key
            edges.append((source, target, cg.weight))
    edges = np.array(edges)
    nodes = np.unique(edges[:, :2])

    # Define graph
    G = nx.DiGraph()

    if weights == "None":
        G.add_edges_from(list(edges[:, :2].astype(int)))
    elif weights == "Binary":
        for e in edges:
            G.add_edge(e[0], e[1], weight=-1 if e[2] < 0.0 else 1)

    for n in nodes:
        G.add_node(n, label=n)

    return G, nodes, edges
