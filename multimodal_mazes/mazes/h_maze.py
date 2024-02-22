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
        assert (self.size % 2) == 1, "Please use an uneven maze size"
        self.maze_type = "HMaze"
        self.goal_channels = []

    def generate(self, number):
        """
        Generates H mazes.
        Arguments:
            number: of mazes to generate
        Generates:
            mazes: see parent class.
            start_locations: see parent class.
            goal_locations: see parent class.
            goal_channels: see above.
            d_maps: see parent class.
            fastest_solutions: see parent class.
        """
        assert (number % 4) == 0, "Please use a number of mazes divisible by 4"

        # Set start locations as the middle
        start_locations = np.repeat(
            [[self.size // 2, self.size // 2]], repeats=number, axis=0
        )

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

        mazes, d_maps, fastest_solutions = [], [], []
        for n in range(number):
            maze = np.zeros(
                shape=(self.size, self.size, self.n_channels + 1), dtype="double"
            )

            # Generate track (1.0) and walls (.0.0)
            maze[:, :, -1] = 0.0
            maze[self.size // 2, 1:-1, -1] = 1.0
            maze[1:-1, [1, -2], -1] = 1.0

            # Set up gradients
            gradient = np.linspace(start=0.1, stop=1.0, num=(self.size - 3))
            gradient_h = gradient[: (len(gradient) // 2)]
            gradient_v = gradient[(len(gradient) // 2) :]

            # Misleading cues
            # Horizontal
            maze[self.size // 2, 1:-1, (1 - goal_channels[n, 0])] = np.concatenate(
                (gradient_h[::-1], np.array(0)[None], gradient_h)
            )

            # Vertical
            maze[1:-1, [1, -2], (1 - goal_channels[n, 1])] = np.repeat(
                np.concatenate((gradient_v[::-1], np.array(0)[None], gradient_v))[
                    :, None
                ],
                repeats=2,
                axis=1,
            )

            # Leading cues
            # Horizontal
            if goal_locations[n, 1] == 1:
                maze[self.size // 2, 1 : (self.size - 1) // 2, goal_channels[n, 0]] = (
                    gradient_h[::-1]
                )
            else:
                maze[self.size // 2, (self.size + 1) // 2 : -1, goal_channels[n, 0]] = (
                    gradient_h
                )

            # Vertical
            if goal_locations[n, 0] == 1:  # up
                maze[
                    ((self.size - 1) // 2) + 1 : -1,
                    np.setdiff1d([1, self.size - 2], goal_locations[n, 1])[0],
                    goal_channels[n, 1],
                ] = gradient_v  # down
                maze[
                    1 : (self.size - 1) // 2, goal_locations[n, 1], goal_channels[n, 1]
                ] = gradient_v[
                    ::-1
                ]  # up
            else:  # down
                maze[
                    ((self.size - 1) // 2) + 1 : -1,
                    goal_locations[n, 1],
                    goal_channels[n, 1],
                ] = gradient_v  # down
                maze[
                    1 : (self.size - 1) // 2,
                    np.setdiff1d([1, self.size - 2], goal_locations[n, 1])[0],
                    goal_channels[n, 1],
                ] = gradient_v[
                    ::-1
                ]  # up

            # Calculate distance map
            d_map = Maze.distance_map(mz=maze, exit=goal_locations[n])
            fastest_solution = d_map[start_locations[n][0], start_locations[n][1]] - 1.0

            # Append to lists
            mazes.append(maze)
            d_maps.append(d_map)
            fastest_solutions.append(fastest_solution)

        self.mazes = mazes
        self.start_locations = start_locations
        self.goal_locations = goal_locations
        self.goal_channels = goal_channels
        self.d_maps = d_maps
        self.fastest_solutions = np.array(fastest_solutions)
