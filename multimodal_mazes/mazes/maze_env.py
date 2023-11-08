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
            goal_locations: a np vector of goal locations.
            maze_type: a string denoting the maze type.
        """
        self.size = size
        self.n_channels = n_channels
        self.mazes = []
        self.goal_locations = []
        self.maze_type = []
