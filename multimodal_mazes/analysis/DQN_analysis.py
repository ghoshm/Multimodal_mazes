# DQN analysis

import numpy as np
import torch
import itertools
import multimodal_mazes
import copy
from torch.autograd.functional import jacobian


def test_dqn_agent(maze_test, agnt, exp_config, noises):
    """
    Test a DQN agent's fitness, input sensitivity and memory
        across multiple noise levels.
    Arguments:
        maze_test: a class containing a set of mazes.
        agnt: an instance of a DQN agent.
        exp_config: a dictionary of hyperparameters.
        noises: a np vector of agent sensor noise levels to test.
    Returns:
        results: a np vector storing the agent's fitness per noise level.
        input_sensitivity: a list of tuples storing the agent's input sensitivity (per noise level).
        memory: a np array where each entry stores the agent's memory per noise level and temporal lag.
    """
    results = np.zeros(len(noises))
    input_sensitivity, memory = [], []

    for a, noise in enumerate(noises):

        # Fitness
        results[a], all_states = multimodal_mazes.eval_fitness(
            genome=None,
            config=None,
            channels=exp_config["channels"],
            sensor_noise_scale=noise,
            drop_connect_p=0.0,
            maze=maze_test,
            n_steps=exp_config["n_steps"],
            agnt=copy.deepcopy(agnt),
            record_states=True,
        )

        # Input sensitivity
        xs, ys = multimodal_mazes.calculate_dqn_input_sensitivity(
            all_states=all_states, agnt=copy.deepcopy(agnt)
        )
        input_sensitivity.append((np.copy(xs), np.copy(ys)))

        # Memory
        mis = multimodal_mazes.calculate_dqn_memory(
            all_states=all_states,
            agnt=copy.deepcopy(agnt),
            n_steps=exp_config["n_steps"],
        )
        memory.append(np.copy(mis))

    return results, input_sensitivity, np.array(memory)


def calculate_dqn_input_sensitivity(all_states, agnt):
    """
    Calulates the network's input sensitivity.
        Defined as the Frobenius norm of it's input-output Jacobian.
    Arguments:
        all_states: a list containing a list per trial; each of which
            contains a tuple per time point of the agent's states.
        agnt: an instance of a DQN agent.
    Returns:
        xs: a numpy array with the directional coherance at each time step.
        ys: a numpy array with the norm of the (input-output) Jacobian at each time step.
    """

    # Reformat data
    all_states = list(itertools.chain.from_iterable(all_states))

    # Define helper function
    def forward(*state):
        """
        Arguments:
            state: a tuple storing tensors of:
                inputs, prev_inputs, hidden, prev_outputs, outputs.
        Returns:
            agnt.outputs: output activations at t.
        """
        agnt.channel_inputs = state[0]
        agnt.outputs, _, _, _ = agnt.forward(
            state[1], state[2], state[3], tensor_input=True
        )

        return agnt.outputs

    # Calculate jacobian norms
    xs, ys = [], []
    for state in all_states:
        jm = jacobian(forward, state)

        net_vector = torch.zeros(2)
        total_input = 0.0

        for a, i in enumerate(np.arange(len(state[0]))[::2]):
            magnitude = state[0][i] + state[0][i + 1]
            direction = torch.tensor([agnt.sensors[0][a], agnt.sensors[1][a]])

            net_vector += magnitude * direction
            total_input += magnitude

        x = (
            torch.norm(net_vector) / total_input
            if total_input > 0
            else torch.tensor(0.0)
        )

        y = torch.norm(jm[0], p="fro")

        xs.append(x)
        ys.append(y)

    # Format data
    xs = np.array(xs)
    ys = np.array(ys)
    xs[np.isnan(xs)] = 0.0

    return xs, ys


def calculate_dqn_memory(all_states, agnt, n_steps):
    """
    Calculate a dqn network's memory.
        Essentially, the (normalised) partial derivative of the input
        w.r.t output, at different temporal lags.
    Arguments:
        all_states: a list containing a list per trial; each of which
            contains a tuple per time point of the agent's states.
        agnt: an instance of a DQN agent.
        n_steps: number of simulation steps.
    Returns:
        tmp: a np vector with the agent's memory per temporal lag.
    """

    # Define helper function
    def forward(*states):
        """
        Arguments:
            states: a tuple storing tensors of:
                inputs, prev_inputs, hidden, prev_outputs, outputs
                from multiple time points.
                I.e. inputs occur every 5th tensor.
        Returns:
            agnt.outputs: output activations at t.
        """

        for t, input in enumerate(states[::5]):

            agnt.channel_inputs = input

            if t == 0:
                agnt.outputs, prev_input, hidden, prev_output = agnt.forward(
                    states[1], states[2], states[3], tensor_input=True
                )
            else:
                agnt.outputs, prev_input, hidden, prev_output = agnt.forward(
                    prev_input, hidden, prev_output, tensor_input=True
                )

        return agnt.outputs

    # Calculate
    memory = [[] for _ in range(n_steps)]

    for a in range(len(all_states)):  # for each trial

        trial_states = tuple(
            itertools.chain.from_iterable(all_states[a])
        )  # a tuple of tensor states

        for b, _ in enumerate(trial_states[::5]):  # for each time point

            jm = jacobian(
                forward, trial_states[: (5 * (b + 1))]
            )  # a tuple of Jacobians (one per state)

            for c, j in enumerate(
                jm[::5][::-1]
            ):  # for each (sensor) input state (from t backwards)
                memory[c].append(
                    torch.norm(j, p="fro") / torch.norm(jm[::5][-1], p="fro")
                )  # append the norm divided by the norm at time t

    # Store
    tmp = []
    for i in range(n_steps):
        memory[i] = np.array(memory[i])
        memory[i][np.isinf(memory[i])] = np.nan
        tmp.append(np.nanmean(memory[i]))

    return np.array(tmp)


def calculate_dqn_w_norms(agnt):
    """
    Calculate the norm of each weight matrix in a DQN model.
    Arguments:
        agnt: an instance of a DQN agent.
    Returns:
        w_norms: a numpy vector - with a norm per weight matrix.
    """

    tmp_f, tmp_o = [], []
    w_norms = np.zeros(9)

    for a, p in enumerate(agnt.parameters()):

        if a <= 1:
            tmp_f.append(torch.norm(p.data, p="fro"))
        else:
            tmp_o.append(torch.norm(p.data, p="fro"))

    w_norms[:2] = np.array(tmp_f)
    w_norms[2:][agnt.wm_flags == 1] = np.array(tmp_o)

    return w_norms


def compute_counterfactual_effects(X, y):
    """
    Compute the counterfactual effect of flipping each
    binary feature in X on the outcome y.

    Parameters:
        X: binary matrix (samples, features).
        y: continuous vector (samples,).

    Returns:
        ce: counterfactual effects for each feature (ce pairs, features).
        f0: indicies without each feature (ce pairs, features).
        f1: indicies with each feature (ce pairs, features).
    """

    n_features = X.shape[1]
    counterfactual_effects = [[] for _ in range(n_features)]
    f0 = [[] for _ in range(n_features)]
    f1 = [[] for _ in range(n_features)]

    for i in range(n_features):

        # Indices where feature i is 0
        indices_0 = np.where(X[:, i] == 0)[0]

        for idx_0 in indices_0:
            # Define counterfactual sample
            counterfactual_sample = X[idx_0].copy()
            counterfactual_sample[i] = 1

            # Find counterfactual sample
            match_idx = np.where((X == counterfactual_sample).all(axis=1))[0][0]

            # Calculate difference
            counterfactual_effects[i].append(y[match_idx] - y[idx_0])
            f0[i].append(idx_0)
            f1[i].append(match_idx)

    return np.array(counterfactual_effects).T, np.array(f0).T, np.array(f1).T
