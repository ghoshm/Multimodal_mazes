# Maze environment

import numpy as np


class Maze:
    def __init__(self, size, n_channels):
        """
        Maze base class.
        Arguments:
            size: the size of the square maze [n].
            n_channels: the number of input channels [n].
        Properties:
            mazes: a list of mazes, each maze is
                np array of size x size x channels + 1.
                Where [:,:,-1] stores the maze structure.
            start_locations: a np array with each start n[r, c].
            goal_locations: a np array with each exit n[r, c].
            maze_type: a string denoting the maze type.
            d_maps: a list storing each mazes distance_map (see function).
            fastest_solutions: a np vector with each mazes fastest solution.
        """
        self.size = size
        self.n_channels = n_channels
        self.mazes = []
        self.start_locations = []
        self.goal_locations = []
        self.maze_type = []
        self.d_maps = []
        self.fastest_solutions = []

    def distance_map(mz, exit):
        """
        Generates a distance map for a given maze.
        Arguments:
            mz: a np array of size x size x channels + 1.
                Where [:,:,-1] stores the maze structure.
            exit: [r, c].
        Returns:
            d_map: a np.array of size x size.
                Which stores each points distance from the exit.
        """
        d_map = np.zeros_like(mz[:, :, -1])  # distance map
        c_map = np.copy(mz[:, :, -1])  # closed map

        open_list = [exit]
        neighbours = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        while open_list:
            cur_rc = open_list.pop(0)

            for neighbour in neighbours:
                new_rc = cur_rc + neighbour

                if c_map[new_rc[0], new_rc[1]] == 0:
                    continue
                else:
                    dist = d_map[cur_rc[0], cur_rc[1]] + 1
                    d_map[new_rc[0], new_rc[1]] = dist
                    open_list.append(new_rc)

            c_map[cur_rc[0], cur_rc[1]] = 0.0

        return d_map
