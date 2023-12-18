# Maze trial

import numpy as np
import multimodal_mazes


def maze_trial(
    mz,
    mz_start_loc,
    mz_goal_loc,
    channels,
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
            location=mz_start_loc, channels=channels, genome=genome, config=config
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
