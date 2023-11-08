# Maze trial

import multimodal_mazes


def maze_trial(mz, mz_goal_loc, channels, genome, config, n_steps):
    """
    Tests a single agent on a single maze.
    Arguments:
        mz: a np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        mz_goal_loc: the location of the goal [x].
        channels: list of active (1) and inative (0) channels e.g. [0,1].
        genome: neat generated genome.
        config: the neat configuration holder.
        n_steps: number of simulation steps.
    Returns:
        time: the number of steps taken to solve the maze.
              Returns n_steps-1 if the agent fails.
        path: a list with the agent's location at each time step [x,y].
    """
    # Reset agent
    agnt = multimodal_mazes.Agent(
        location=[5, 5], channels=channels, genome=genome, config=config
    )

    path = [list(agnt.location)]
    # Sensation-action loop
    for time in range(n_steps):
        agnt.sense(mz)
        agnt.act(mz)

        path.append(list(agnt.location))
        # If the end is reached
        if agnt.location[1] == mz_goal_loc:
            break

    return time, path  # returning a class is more flexible
