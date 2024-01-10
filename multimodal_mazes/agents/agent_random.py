# Random baseline agent

import numpy as np
from multimodal_mazes.agents.agent import Agent


class AgentRandom(Agent):
    def __init__(self, location, channels):
        super().__init__(location, channels)
        self.type = "Random"
        """
        Creates a random agent for baseline comparisons. 
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

    def policy(self):
        """
        Assign a value to each action.
        Policy is a random choice among actions, ignoring channel inputs.
        """

        self.outputs = np.zeros(len(self.actions[0]))
        self.outputs += np.random.rand(len(self.outputs))
