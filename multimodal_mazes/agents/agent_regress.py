# Regression-based agent

import numpy as np
from multimodal_mazes.agents.agent import Agent
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class AgentRegress(Agent):
    def __init__(self, location, channels, degree):
        super().__init__(location, channels)
        self.type = "AgentRegress"
        self.degree = degree
        self.reg = []
        self.poly = []

        """
        Creates a regression-based agent. 
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

    def generate_policy(self, maze):
        """ """
        # Create training data
        x, y = [], []
        for n in range(len(maze.mazes)):
            mz = maze.mazes[n]
            d_map = np.copy(maze.d_maps[n])
            d_map[d_map == 0.0] = np.inf
            d_map[maze.goal_locations[n][0], maze.goal_locations[n][1]] = 0.0

            x, y = [], []
            self.location = np.copy(maze.start_locations[n])
            while (self.location != maze.goal_locations[n]).any():
                # Record senses
                self.sense(mz)
                x.append(list(self.channel_inputs.reshape(-1)))

                # Record best action
                action_dists = d_map[
                    self.location[0] + self.actions[0],
                    self.location[1] + self.actions[1],
                ]
                action = np.argmin(
                    action_dists
                )  # consider adding random noise to avoid argmin bias?
                y.append(action)

                # Take action
                self.location += [self.actions[0][action], self.actions[1][action]]

        x = np.array(x)
        y = np.array(y)

        # # OLD Create training data
        # x, y = [], []
        # for n in range(len(maze.mazes)):
        #     mz = maze.mazes[n]
        #     mz_start_loc = maze.start_locations[n]
        #     mz_goal_loc = maze.goal_locations[n]
        #     d_map = np.copy(maze.d_maps[n])
        #     d_map[d_map > d_map[mz_start_loc[0], mz_start_loc[1]]] = 0.0
        #     d_map[d_map == 0.0] = np.inf
        #     d_map[mz_goal_loc[0], mz_goal_loc[1]] = 0.0

        #     for rc in zip(np.where(d_map != np.inf)[0], np.where(d_map != np.inf)[1]):
        #         self.location = np.array(rc)  # set the agent's location

        #         if np.array_equal(self.location, mz_goal_loc):
        #             continue
        #         else:
        #             # Record senses
        #             self.sense(mz)
        #             x.append(list(self.channel_inputs.reshape(-1)))

        #             # Record best action
        #             action_dists = d_map[
        #                 rc[0] + self.actions[0], rc[1] + self.actions[1]
        #             ]
        #             # action_dists += np.random.rand(len(action_dists)) / 1000
        #             y.append(np.argmin(action_dists))

        # x = np.array(x)
        # y = np.array(y)

        # Check label consistency
        tmp = []
        for n in range(x.shape[0]):
            idx = np.where((x == x[n]).all(axis=1))[0]

            if len(np.unique(y[idx])) >= 2:
                print("Inconsistent training labels")

        # Regression
        poly = PolynomialFeatures(
            degree=self.degree, interaction_only=False, include_bias=True
        )
        x_poly = poly.fit_transform(x)
        reg = LinearRegression().fit(x_poly, y)

        self.reg = reg
        self.poly = poly

    def policy(self):
        """
        Assign a value to each action.
        AgentRegress policy is a
        """
        x = self.channel_inputs.reshape(1, -1)
        p = self.reg.predict(self.poly.transform(x))

        self.outputs[:] = np.zeros(len(self.actions[0]))
        self.outputs[int(np.round(p))] = 1.0
