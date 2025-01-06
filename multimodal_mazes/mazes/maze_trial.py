# Maze trial

import numpy as np
import multimodal_mazes
import torch


def maze_trial(
    mz,
    mz_start_loc,
    mz_goal_loc,
    channels,
    sensor_noise_scale,
    drop_connect_p,
    n_steps,
    agnt=None,
    genome=None,
    config=None,
    record_states=False,
):
    """
    Tests a single agent on a single maze.
    Arguments:
        mz: a np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        mz_start_loc: the location of the start [r,c].
        mz_goal_loc: the location of the goal [r,c].
        channels: list of active (1) and inative (0) channels e.g. [0,1].
        sensor_noise_scale: the scale of the noise applied to every sensor.
        drop_connect_p: the probability of edge drop out, per time step.
        genome: neat generated genome.
        config: the neat configuration holder.
        n_steps: number of simulation steps.
    Returns:
        time: the number of steps taken to solve the maze.
              Returns n_steps-1 if the agent fails.
        path: a list with the agent's location at each time step [r,c].
        states: a list of the agent's states at each time step.
            Each entry is a tuple storing tensors of:
            inputs, prev_inputs, hidden, prev_outputs, outputs.
    """
    # Reset agent
    if agnt is None:
        agnt = multimodal_mazes.AgentNeat(
            location=mz_start_loc,
            channels=channels,
            sensor_noise_scale=sensor_noise_scale,
            drop_connect_p=drop_connect_p,
            genome=genome,
            config=config,
        )

    elif agnt.type == "AgentDQN":
        agnt.prev_input = torch.zeros(agnt.n_input_units)
        agnt.hidden = torch.zeros(agnt.n_hidden_units)
        agnt.prev_output = torch.zeros(agnt.n_output_units)

        agnt.location = np.copy(mz_start_loc)
        agnt.outputs = torch.zeros(agnt.n_output_units)
        agnt.sensor_noise_scale = sensor_noise_scale

        if record_states:
            states = []

    else:
        agnt.location = np.copy(mz_start_loc)
        agnt.sensor_noise_scale = sensor_noise_scale
        agnt.outputs *= 0.0
        if agnt.type == "Hidden skip":
            agnt.memory = np.zeros_like(agnt.outputs)
        elif agnt.type == "Levy":
            agnt.flight_length = 0
            agnt.collision = 0

    path = [list(agnt.location)]
    # Sensation-action loop
    for time in range(n_steps):
        agnt.sense(mz)
        if record_states:
            states.append(
                (
                    torch.from_numpy(agnt.channel_inputs.reshape(-1))
                    .to(torch.float32)
                    .requires_grad_(False),
                    agnt.prev_input.detach().clone(),
                    agnt.hidden.detach().clone(),
                    agnt.prev_output.detach().clone(),
                )
            )
        agnt.policy()
        if record_states:
            states[-1] += tuple([agnt.outputs.detach().clone()])
        agnt.act(mz)

        path.append(list(agnt.location))
        # If the exit is reached
        if np.array_equal(agnt.location, mz_goal_loc):
            break

    if record_states == False:
        return time, path
    elif record_states == True:
        return time, path, states


def eval_fitness(
    genome,
    config,
    channels,
    sensor_noise_scale,
    drop_connect_p,
    maze,
    n_steps,
    agnt=None,
):
    """
    Evalutes the fitness of the provided genome across a set of mazes.
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
        channels: list of active (1) and inative (0) channels e.g. [0,1].
        sensor_noise_scale: the scale of the noise applied to every sensor.
        drop_connect_p: the probability of edge drop out, per time step.
        maze: a class containing a set of mazes.
        n_steps: the max number of simulation steps per maze.
    Returns:
        fitness: the mean fitness across mazes, between [0,1].
    """
    fitness, times, paths = [], [], []
    # For each maze
    for mz_n, mz in enumerate(maze.mazes):
        # Run trial
        time, path = multimodal_mazes.maze_trial(
            mz=mz,
            mz_start_loc=maze.start_locations[mz_n],
            mz_goal_loc=maze.goal_locations[mz_n],
            channels=channels,
            sensor_noise_scale=sensor_noise_scale,
            drop_connect_p=drop_connect_p,
            n_steps=n_steps,
            agnt=agnt,
            genome=genome,
            config=config,
        )

        # Record normalised fitness
        times.append(
            1
            - (
                (time - maze.fastest_solutions[mz_n])
                / (n_steps - 1 - maze.fastest_solutions[mz_n])
            )
        )

        paths.append(
            (maze.d_maps[mz_n].max() - maze.d_maps[mz_n][path[-1][0], path[-1][1]])
            / maze.d_maps[mz_n].max()
        )

    # Fitness
    fitness = (np.array(times) + np.array(paths)) * 0.5

    # Return fitness
    return np.array(fitness).mean()


def id_top_agents(fitness_cutoff, exp_data, maze, exp_config, genomes, config):
    """
    Returns a list of agents with both evolutionary (training) and test
        fitness >= fitness_cutoff.
    Arguments:
        fitness_cutoff: the cuttoff value to use (between 0 and 1).
        exp_data: a structured np array of agents x info (e.g. fitness).
        maze: a class containing a set of mazes.
        exp_config: a dict with the exp configuration.
        genomes: neat generated genomes.
        config: the neat configuration holder.
    Returns:
        top_agents: agents: a list of indicies of top agents / genomes.
    """

    top_agents = []
    for g in np.where(exp_data["fitness"] >= fitness_cutoff)[0]:
        _, genome, channels = genomes[g]
        genome = multimodal_mazes.prune_architecture(genome, config)
        fitness = multimodal_mazes.eval_fitness(
            genome=genome,
            config=config,
            channels=channels,
            sensor_noise_scale=exp_config["sensor_noise_scale"],
            drop_connect_p=exp_config["drop_connect_p"],
            maze=maze,
            n_steps=exp_config["n_steps"],
        )

        if fitness >= fitness_cutoff:
            top_agents.append(g)

    return top_agents
