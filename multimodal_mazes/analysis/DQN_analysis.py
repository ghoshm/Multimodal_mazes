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
        xs: a numpy array with the left minus right signal at each time step.
        ys: a numpy array with the norm of the (input-output) Jacobian.
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
        x = (torch.sum(state[0][[0, 1]]) - torch.sum(state[0][[2, 3]])) / torch.sum(
            state[0][:4]
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
