# Maze trial

import numpy as np
import multimodal_mazes


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
    else:
        agnt.location = np.copy(mz_start_loc)

    path = [list(agnt.location)]
    # Sensation-action loop
    for time in range(n_steps):
        agnt.sense(mz)
        agnt.policy()
        agnt.act(mz)

        path.append(list(agnt.location))
        # If the exit is reached
        if np.array_equal(agnt.location, mz_goal_loc):
            break

    return time, path  # returning a class would be more flexible


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
