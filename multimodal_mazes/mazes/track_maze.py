# Track maze
import numpy as np
from multimodal_mazes.mazes.maze_env import Maze


class TrackMaze(Maze):
    """
    Track maze class.
    Additional properties:
        goal_channels: a np vector with each trials leading channel.
    """

    def __init__(self, size, n_channels):
        super().__init__(size, n_channels)
        assert (self.size % 2) == 1, "Please use an uneven maze size"
        self.maze_type = "TrackMaze"
        self.goal_channels = []

    def generate(self, number, noise_scale, gaps):
        """
        Generates track mazes.
        Arguments:
            number: of mazes to generate.
            noise_scale: scale of gaussian noise.
            gaps: add sensory gaps or not [True, False].
        Generates:
            mazes: see parent class.
            goal_locations: parent class.
            goal_channels: see above.
            fastest_solutions: see parent class.
        """
        assert (number % 2) == 0, "Please use an even number of mazes"

        # Set start locations as the middle
        start_locations = np.repeat(
            [[self.size // 2, self.size // 2]], repeats=number, axis=0
        )

        # Set goal locations as the middle row (self.size // 2) and
        # either the left (1) or right (size - 2) column
        goal_locations = np.repeat(
            [[self.size // 2, 1], [self.size // 2, self.size - 2]],
            repeats=number // 2,
            axis=0,
        )

        # Set which channel encodes the goal
        goal_channels = np.repeat([0, 1], repeats=number // 2)
        np.random.shuffle(goal_channels)

        mazes, d_maps, fastest_solutions = [], [], []
        for n in range(number):
            maze = np.zeros(
                shape=(self.size, self.size, self.n_channels + 1), dtype="double"
            )

            # Generate track (1.0) and walls (0.0)
            maze[:, :, -1] = 0.0
            maze[self.size // 2, 1:-1, -1] = 1.0

            # Set up gradients
            gradient = np.linspace(start=0.1, stop=0.5, num=((self.size - 2) // 2))

            # Cues
            if goal_locations[n, 1] == 1:  # left
                # Leading
                maze[self.size // 2, 1 : (self.size - 1) // 2, goal_channels[n]] = (
                    gradient[::-1]
                )

                # Misleading
                maze[self.size // 2, 1:-1, (1 - goal_channels[n])] = np.concatenate(
                    (gradient[::-1], np.array(0)[None], gradient * 1)
                )

            else:  # Right
                # Leading
                maze[self.size // 2, (self.size + 1) // 2 : -1, goal_channels[n]] = (
                    gradient
                )

                # Misleading
                maze[self.size // 2, 1:-1, (1 - goal_channels[n])] = np.concatenate(
                    (gradient[::-1] * 1, np.array(0)[None], gradient)
                )

            # Noise
            r, c = np.where(maze[:, :, -1])
            maze[r, c, :-1] += np.random.normal(
                loc=0.0, scale=noise_scale, size=(len(r), (maze.shape[2] - 1))
            )
            maze = np.clip(maze, a_min=0.0, a_max=None)

            # Gaps
            if gaps:
                if goal_locations[n, 1] == 1:  # left
                    maze[goal_locations[n, 0], goal_locations[n, 1] + 1, :2] = 0.0
                else:
                    maze[goal_locations[n, 0], goal_locations[n, 1] - 1, :2] = 0.0

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
