# H maze
import numpy as np
from multimodal_mazes.mazes.maze_env import Maze


class HMaze(Maze):
    """
    H maze class.
    Additional properties:
        goal_channels: np array which stores
            how the channels encode the goal n[first lead, second lead]
    """

    def __init__(self, size, n_channels):
        super().__init__(size, n_channels)
        self.maze_type = "HMaze"
        self.goal_channels = []

    def generate(self, number):
        """
        Generates H mazes.
        Arguments:
            number: of mazes to generate
        Generates:
            mazes: see parent class.
            goal_locations: see parent class.
            goal_channels: see above.
            fastest_solutions: see parent class.
        """
        assert (number % 4) == 0, "Please use a number of mazes divisible by 4"

        # Set goal locations as n[r, c]
        goal_locations = np.repeat(
            [
                [1, 1],
                [1, self.size - 2],
                [self.size - 2, 1],
                [self.size - 2, self.size - 2],
            ],
            repeats=number // 4,
            axis=0,
        )

        # Set how the channels encode the goal n[first lead, second lead]
        goal_channels = np.repeat([[0, 1], [1, 0]], repeats=number // 2, axis=0)
        np.random.shuffle(goal_channels)

        mazes = []
        for n in range(number):
            maze = np.zeros(
                shape=(self.size, self.size, self.n_channels + 1), dtype="double"
            )

            # Generate track (1.0) and walls (.0.0)
            maze[:, :, -1] = 0.0
            maze[self.size // 2, 1:-1, -1] = 1.0
            maze[1:-1, [1, -2], -1] = 1.0

            # Set up gradients
            gradient = np.linspace(start=0.1, stop=1.0, num=((self.size - 2) // 2))

            # Misleading cues
            # Horizontal
            maze[self.size // 2, 1:-1, (1 - goal_channels[n, 0])] = np.concatenate(
                (gradient[::-1], np.array(0)[None], gradient)
            )

            # Vertical
            maze[1:-1, [1, -2], (1 - goal_channels[n, 1])] = np.repeat(
                np.concatenate((gradient[::-1], np.array(0)[None], gradient))[:, None],
                repeats=2,
                axis=1,
            )

            # Leading cues
            # Horizontal
            if goal_locations[n, 1] == 1:
                maze[
                    self.size // 2, 1 : (self.size - 1) // 2, goal_channels[n, 0]
                ] = gradient[::-1]
            else:
                maze[
                    self.size // 2, (self.size + 1) // 2 : -1, goal_channels[n, 0]
                ] = gradient

            # Vertical
            if goal_locations[n, 0] == 1:  # up
                maze[
                    ((self.size - 1) // 2) + 1 : -1,
                    np.setdiff1d([1, self.size - 2], goal_locations[n, 1])[0],
                    goal_channels[n, 1],
                ] = gradient  # down
                maze[
                    1 : (self.size - 1) // 2, goal_locations[n, 1], goal_channels[n, 1]
                ] = gradient[
                    ::-1
                ]  # up
            else:  # down
                maze[
                    ((self.size - 1) // 2) + 1 : -1,
                    goal_locations[n, 1],
                    goal_channels[n, 1],
                ] = gradient  # down
                maze[
                    1 : (self.size - 1) // 2,
                    np.setdiff1d([1, self.size - 2], goal_locations[n, 1])[0],
                    goal_channels[n, 1],
                ] = gradient[
                    ::-1
                ]  # up

            # Append to list
            mazes.append(maze)

        self.mazes = mazes
        self.goal_locations = goal_locations
        self.goal_channels = goal_channels
        self.fastest_solutions = np.repeat(self.size - 4, repeats=number)
