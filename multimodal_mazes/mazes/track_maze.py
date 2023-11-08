# Track maze
import numpy as np
from multimodal_mazes.mazes.maze_env import Maze


class TrackMaze(Maze):
    """
    Track maze class.
    Additional properties:
        goal_channels: a np vector of goal locations.
    """

    def __init__(self, size, n_channels):
        super().__init__(size, n_channels)
        self.maze_type = "TrackMaze"
        self.goal_channels = []

    def generate(self, number):
        """
        Generates track mazes.
        Arguments:
            number: of mazes to generate
        Generates:
            mazes: see parent class.
            goal_locations: parent class.
            goal_channels: see above.
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
