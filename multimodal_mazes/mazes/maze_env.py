# Maze environment

import numpy as np
from multimodal_mazes.agents.agent import Agent
import multimodal_mazes


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

    def shortest_path(mz, d_map, start, exit):
        """
        Returns the shortest path (start-exit) through a given maze.
        Arguments:
            mz: a np array of size x size x channels + 1.
                Where [:,:,-1] stores the maze structure.
            d_map: a np.array of size x size.
                Which stores each points distance from the exit.
            start: [r, c].
            exit: [r, c].
        Returns:
            path: a list with the shortest path positions [r,c].
                Inc the start and exit.
        """

        d_map_inv = (d_map.max() - d_map) / d_map.max()
        d_map_inv = d_map_inv * mz[:, :, -1]
        agnt = Agent(location=start, channels=[1, 1])

        path = [list(agnt.location)]
        while (agnt.location != exit).any():

            # Record best action
            action_dists = d_map_inv[
                agnt.location[0] + agnt.actions[0],
                agnt.location[1] + agnt.actions[1],
            ]

            action = np.argmax(action_dists)

            # Take action
            agnt.location += [agnt.actions[0][action], agnt.actions[1][action]]
            path.append(list(agnt.location))

        return path

    def generate_sensation_action_pairs(self, sensor_noise_scale):
        """
        Generate channel_input-action pairs for every position in each maze.
        Arguments:
            sensor_noise_scale: the scale of the noise applied to every sensor.
        Creates:
            channel_inputs: inputs from locations (sensors x channels x batch).
            ci_actions: the correct action for each channel_input (np vector).
        """
        # Setup
        channel_inputs, actions = [], []
        agnt = Agent(location=None, channels=[1, 1])
        agnt.sensor_noise_scale = sensor_noise_scale

        # For each maze, and every position
        for n in range(len(self.mazes)):
            mz = self.mazes[n]
            d_map_inv = (self.d_maps[n].max() - self.d_maps[n]) / self.d_maps[n].max()
            d_map_inv = d_map_inv * mz[:, :, -1]

            rcs = np.argwhere(mz[:, :, -1])
            for _, rc in enumerate(rcs):
                if (rc != self.goal_locations[n]).any():

                    # Record channel data
                    agnt.location = np.copy(rc)
                    agnt.sense(mz)
                    channel_inputs.append(np.copy(agnt.channel_inputs))

                    # Record best action
                    action_dists = d_map_inv[
                        agnt.location[0] + agnt.actions[0],
                        agnt.location[1] + agnt.actions[1],
                    ]

                    actions.append(np.argmax(action_dists))

        # Store
        self.channel_inputs = np.stack(
            channel_inputs, axis=2
        )  # (sensors x channels x batch)
        self.ci_actions = np.array(actions)  # vector


def maze_generator_wrapper(exp_config):
    """
    Generate a set of mazes from an exp_config file.
    Arguments:
        exp_config: loaded hyperparameters.
    Returns:
        mazes: a set of mazes.
        maze_test: a set of 1000 mazes.
    """

    if exp_config["maze_type"] == "Track":
        maze = multimodal_mazes.TrackMaze(
            size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
        )
        maze.generate(
            number=exp_config["n_mazes"],
            noise_scale=exp_config["maze_noise_scale"],
            gaps=exp_config["maze_gaps"],
        )

        maze_test = multimodal_mazes.TrackMaze(
            size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
        )
        maze_test.generate(
            number=1000,
            noise_scale=exp_config["maze_noise_scale"],
            gaps=exp_config["maze_gaps"],
        )

    elif exp_config["maze_type"] == "H":
        maze = multimodal_mazes.HMaze(
            size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
        )
        maze.generate(number=exp_config["n_mazes"], gaps=exp_config["maze_gaps"])

        maze_test = multimodal_mazes.HMaze(
            size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
        )
        maze_test.generate(number=1000, gaps=exp_config["maze_gaps"])

    return maze, maze_test
