# Agent

import numpy as np


class Agent:
    def __init__(self, location, channels):
        """
        Creates a new agent.
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
        Properties:
            fitness: the agents mean fitness across mazes, between [0,1].
            actions: possible movements in [r,c].
            sensors: inputs in [r,c].
            channel_inputs: the agents inputs from it's location (sensors x channels).
            outputs: the policy assigned value for each action.
        """
        self.location = np.array(location)
        self.channels = np.array(channels)
        self.fitness = []
        self.actions = [[0, 0, -1, 1], [-1, 1, 0, 0]]
        self.sensors = [[0, 0, -1, 1], [-1, 1, 0, 0]]
        self.channel_inputs = np.zeros((len(self.sensors[0]), len(self.channels)))
        self.outputs = np.zeros(len(self.actions[0]))
        self.type = []

    def sense(self, env):
        """
        Generates channel data for agent.
        Arguments:
            env: a np array of size x size x channels + 1.
                Where [:,:,-1] stores the maze structure.
        Updates:
            self.channel_inputs: a sensors x channels np array.
            All agents will have max channels, but inactive ones will be zeroed.
        """
        # Generate channel data
        rc = [self.location[0] + self.sensors[0], self.location[1] + self.sensors[1]]

        self.channel_inputs[:] = env[rc[0], rc[1], :-1]

        # Zero out inactive channels
        self.channel_inputs *= self.channels

    def act(self, env):
        """
        Updates the agent's state by one action.
        Arguments:
            env: a np array of size x size x channels + 1.
                Where [:,:,-1] stores the maze structure.
        Updates:
            self.location: by one action.
            If the action collides with a wall it is ignored.
        """

        # Choose action
        action = np.argmax(self.outputs)

        # Act (if the action does not collide with a wall)
        if (
            env[
                self.location[0] + self.actions[0][action],
                self.location[1] + self.actions[1][action],
                -1,
            ]
            == 1.0
        ):
            self.location += [self.actions[0][action], self.actions[1][action]]
