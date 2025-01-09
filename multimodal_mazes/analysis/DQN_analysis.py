# DQN analysis

import numpy as np
import torch
import itertools
import multimodal_mazes
import copy
from torch.autograd.functional import jacobian
from sklearn.feature_selection import mutual_info_regression


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
        mis = multimodal_mazes.estimate_dqn_memory(
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


def estimate_dqn_memory(all_states, agnt, n_steps):
    """
    Estimates a networks memory.
        Defined as the norm of the estimated mutual information
        between it's inputs and outputs at different time lags.
    Arguments:
        all_states: a list containing a list per trial; each of which
            contains a tuple per time point of the agent's states.
        agnt: an instance of a DQN agent.
        n_steps: number of simulation steps.
    Returns:
        mis: a np vector with the agent's memory per temporal lag.
    """

    # Reorganise states by time
    x, y, t = [], [], []
    for trial_states in all_states:
        for time in range(n_steps):

            try:
                x.append(trial_states[time][0])  # input at time
                y.append(trial_states[time][-1])  # output at time
            except:
                x.append(np.zeros(agnt.n_input_units) * np.nan)
                y.append(np.zeros(agnt.n_output_units) * np.nan)

            t.append(time)

    x, y, t = np.array(x), np.array(y), np.array(t)

    # Estimate MI at different lags
    mis = []
    for lag in range(n_steps):
        x_tmp = x[t <= (n_steps - 1 - lag)]
        y_tmp = y[t >= lag]

        x_tmp = x_tmp[np.isnan(y_tmp).all(1) == 0]
        y_tmp = y_tmp[np.isnan(y_tmp).all(1) == 0]

        if x_tmp.any():
            mi = torch.tensor(
                np.array(
                    [
                        mutual_info_regression(x_tmp, y_tmp[:, i], n_neighbors=3)
                        for i in range(y_tmp.shape[1])
                    ]
                )
            )
            mis.append(torch.norm(mi, p="fro"))
        else:
            mis.append(np.nan)

    return np.array(mis)
