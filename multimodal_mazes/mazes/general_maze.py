# General maze
import numpy as np
from multimodal_mazes.mazes.maze_env import Maze
from sklearn.metrics import DistanceMetric
import copy


class GeneralMaze(Maze):
    """
    General maze class.
    Additional properties:
    """

    def __init__(self, size, n_channels):
        super().__init__(size, n_channels)
        assert (self.size % 2) == 1, "Please use an uneven maze size"
        self.maze_type = "GeneralMaze"

    def generate(self, number, noise_scale):
        """
        Generates general mazes.
        Arguments:
            number: of mazes to generate.
            noise_scale: scale of gaussian noise added to cues.
        Generates:
            mazes: see parent class.
            stat_locations: parent class.
            goal_locations: parent class.
            d_maps: parent class.
            fastest_solutions: parent class.
        """
        corners = np.array(
            [
                [1, 1],
                [1, self.size - 2],
                [self.size - 2, 1],
                [self.size - 2, self.size - 2],
            ]
        )

        mazes, start_locations, goal_locations, d_maps, fastest_solutions = (
            [],
            [],
            [],
            [],
            [],
        )

        for _ in range(number):
            maze = np.zeros(
                shape=(self.size, self.size, self.n_channels + 1), dtype="double"
            )

            # Generate maze structure
            maze[:, :, -1] = aldous_broder(self.size)

            # Choose start and goal locations
            pos_s_g = corners[np.where(maze[corners[:, 0], corners[:, 1], -1])[0], :]
            locs = np.random.choice(len(pos_s_g), size=2, replace=False)
            start_loc = pos_s_g[locs[0]]
            goal_loc = pos_s_g[locs[1]]

            # Calculate distance map
            d_map = Maze.distance_map(mz=maze, exit=goal_loc)

            # Calculate shortest path
            path = Maze.shortest_path(
                mz=maze,
                d_map=d_map,
                start=start_loc,
                exit=goal_loc,
            )

            # Fill sensory cues
            maze = shortest_path_fill(mz=maze, channels=[0], d_map=d_map)
            # maze = random_fill(mz=maze, channels=[1])
            maze = shortest_path_fill(mz=maze, channels=[1], d_map=d_map)
            # maze = distance_fill(
            #     mz=maze, channels=[1], exit=goal_loc, distance="chebyshev"
            # )

            # Divide multimodal cue values
            maze[:, :, :-1] /= maze.shape[2] - 1

            # Noise
            r, c = np.where(maze[:, :, -1])
            maze[r, c, :-1] += np.random.normal(
                loc=0.0, scale=noise_scale, size=(len(r), (maze.shape[2] - 1))
            )
            maze = np.clip(maze, a_min=0.0, a_max=1.0)

            # Append to lists
            mazes.append(maze)
            start_locations.append(start_loc)
            goal_locations.append(goal_loc)
            d_maps.append(d_map)
            fastest_solutions.append(len(path) - 2)

        self.mazes = mazes
        self.start_locations = start_locations
        self.goal_locations = goal_locations
        self.d_maps = d_maps
        self.fastest_solutions = np.array(fastest_solutions)


def aldous_broder(size):
    """
    Generates the structure of a single maze using the Aldous-Broder algorithm.
    See mazes for programmers page 55.
    Arguments:
        size: the size of the square maze [n].
    Returns:
        maze: a np array of size x size with the maze structure.
            Track (1.0) and walls (0.0).
    """

    maze = np.zeros((size, size), dtype=int)
    neighbours = np.array([[0, -2], [0, 2], [-2, 0], [2, 0]], dtype=int)
    inner_grid = np.linspace(start=1, stop=(size - 2), num=(size - 2), dtype=int)[::2]

    cell = [
        np.random.choice(inner_grid),
        np.random.choice(inner_grid),
    ]  # choose a random starting cell
    unvisited = (len(inner_grid) ** 2) - 1

    # Loop
    while unvisited > 0:
        maze[cell[0], cell[1]] = 1  # fill current cell

        adjacent_cells = cell + neighbours  # all adjacent cells
        allowed_moves = neighbours[
            ((adjacent_cells >= 1) & (adjacent_cells <= (size - 2))).all(1), :
        ]  # define allowed moves

        move = allowed_moves[np.random.choice(len(allowed_moves))]  # choose a move
        new_cell = cell + move  # move to new cell

        if maze[new_cell[0], new_cell[1]] == 0:  # if this new cell is unvisited
            in_between = cell + (move / 2).astype(int)  # link the two cells
            maze[in_between[0], in_between[1]] = 1
            unvisited -= 1

        cell = np.copy(new_cell)

    return maze


def shortest_path_fill(mz, channels, d_map):
    """
    Fill sensory cues into a single maze,
        using the shortest path to the exit.
    Arguments:
        mz: a np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        channels: a list denoting which channels to fill.
        d_map: a np.array of size x size.
            Which stores each points distance from the exit.
    Returns:
        mz: with sensory cues filled into the chosen channels.
    """

    # Set each positions sensory data
    for ch in channels:
        mz[:, :, ch] = (d_map.max() - d_map) / d_map.max()
        mz[:, :, ch] *= mz[:, :, -1]

    return mz


def random_fill(mz, channels):
    """
    Randomly fill sensory cues into a single maze.
    Arguments:
        mz: a np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        channels: a list denoting which channels to fill.
    Returns:
        mz: with sensory cues filled into the chosen channels.
    """

    # Set each positions sensory data
    for ch in channels:
        mz[:, :, ch] = np.random.rand(mz.shape[0], mz.shape[1])
        mz[:, :, ch] *= mz[:, :, -1]

    return mz


def distance_fill(mz, channels, exit, distance):
    """
    Fill sensory cues into a single maze,
        using the distance to the exit.
    Arguments:
        mz: a np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        channels: a list denoting which channels to fill.
        exit: the maze exit [r, c].
        distance: a metric implemented by sklearn.
    Returns:
        mz: with sensory cues filled into the chosen channels.
    """
    dist = DistanceMetric.get_metric(distance)
    rc = np.argwhere(mz[:, :, -1] == 1)
    rc_dist = dist.pairwise(rc.tolist(), [exit.tolist()]).reshape(-1)

    # Set each positions sensory data
    for ch in channels:
        mz[rc[:, 0], rc[:, 1], ch] = (rc_dist.max() - rc_dist) / rc_dist.max()

    return mz


def path_fidelity_fill(mz, d_map, path, path_fidelity):
    """
    Fill sensory cues into a single maze,
        based on the path fidelity.
    Arguments:
        mz: a np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        d_map: a np.array of size x size.
            Which stores each points distance from the exit.
        path: a list with the shortest path positions [r,c].
            Inc the start and exit.
        path_fidelity: the ratio of positions on:off the shortest path,
            with cues in both channels:
                0 - unimodal cues (from channel 0) on the path, multimodal off.
                1 - bimodal cues on the path, unimodal off
    Returns:
        mz: with the sensory cues filled in.
    """

    # Set each positions sensory data depending on its distance from the exit
    for ch in range(mz.shape[2] - 1):
        mz[:, :, ch] = (d_map.max() - d_map) / d_map.max()
        mz[:, :, ch] *= mz[:, :, -1]

    # On the shortest path
    idx = np.random.rand(len(path)) <= (1 - path_fidelity)
    mz[np.array(path)[idx, 0], np.array(path)[idx, 1], 1] = 0.0

    # Off the shortest path
    track = np.vstack(np.where(mz[:, :, -1])).T  # positions [r,c]
    off_path = []
    for t in track:
        if not any(np.equal(path, t).all(1)):
            off_path.append(t)

    # remove all cues
    off_path = np.array(off_path)  # positions [r,c]
    mz[
        off_path[:, 0],
        off_path[:, 1],
        :-1,
    ] = 0.0

    # # set some cues to be unimodal (depending on path_fidelity)
    # idx = np.random.rand(len(off_path)) <= path_fidelity
    # mz[
    #     off_path[idx, 0],
    #     off_path[idx, 1],
    #     np.random.choice([0, 1], size=sum(idx)),
    # ] = 0.0

    # Halve multimodal cue values
    ms_r, ms_c = np.where((mz[:, :, 0] > 0) & (mz[:, :, 1] > 0))
    mz[ms_r, ms_c, :2] /= 2

    return mz


def sparse_cues(maze, sparsity):
    """
    Remove cues from a set of mazes.
    Arguments:
        maze: a list of mazes, each maze is
            np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        sparsity: the fraction of cues to remove [0,1].
            0 - return a dense maze (with all cues).
            1 - remove all cues.
    Returns:
        maze_sparse: maze with some fraction of
            sensory cues removed from each maze.
    """
    maze_sparse = copy.deepcopy(maze)

    for a, mz_s in enumerate(maze_sparse.mazes):
        track = np.vstack(np.where(mz_s[:, :, -1])).T  # positions [r,c]
        idx = np.random.rand(len(track)) <= sparsity
        maze_sparse.mazes[a][track[idx, 0], track[idx, 1], :-1] = 0.0

    return maze_sparse
