# Maze environment

import numpy as np

# Maze as a parent class which other mazes inherit from


class Maze:
    def __init__(self, size, n_channels):
        """
        Maze object.
        Arguments:
            size: the size of the square maze [n].
            n_channels: the number of input channels [n].
        """
        self.size = size
        self.n_channels = n_channels
        self.mazes = []
        self.goal_locations = []
        self.goal_channels = []
        self.maze_type = []

    def generate_track_mazes(self, number):
        """
        Generates track mazes.
        Arguments:
            number: of mazes to generate
        Returns:
            mazes: a number length list of mazes, each maze
            is a np array of size x size x channels + 1.
            Where [:,:,-1] stores the maze structure.
        """
        # Set goal locations as left (1) or right (size - 2)
        goal_locations = np.repeat([1, self.size - 2], repeats=number // 2)

        # Set which channel encodes the goal
        goal_channels = np.repeat([0, 1], repeats=number // 2)
        np.random.shuffle(goal_channels)

        mazes = []
        for n in range(number):
            maze = np.zeros(
                shape=(self.size, self.size, self.n_channels + 1), dtype="double"
            )

            # Generate track (0.) and walls  (1.)
            maze[:, :, -1] = 1.0
            maze[self.size // 2, 1:-1, -1] = 0.0

            # Set up gradients
            gradient = np.linspace(start=0.1, stop=1.0, num=((self.size - 2) // 2))

            # Misleading channel
            maze[self.size // 2, 1:-1, (1 - goal_channels[n])] = np.concatenate(
                (gradient[::-1], np.array(0)[None], gradient)
            )

            # Leading channel
            if goal_locations[n] == 1:
                maze[
                    self.size // 2, 1 : (self.size - 1) // 2, goal_channels[n]
                ] = gradient[::-1]
            else:
                maze[
                    self.size // 2, (self.size + 1) // 2 : -1, goal_channels[n]
                ] = gradient

            # Append to list
            mazes.append(maze)

        self.mazes = mazes
        self.goal_locations = goal_locations
        self.goal_channels = goal_channels
        self.maze_type = "Track"
