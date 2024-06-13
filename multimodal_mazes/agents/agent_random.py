# Random baseline agent

import numpy as np
from multimodal_mazes.agents.agent import Agent


class AgentRandom(Agent):
    def __init__(self, location, channels, motion):
        super().__init__(location, channels)
        self.type = motion

        if self.type == "Levy":
            self.flight_length = 0  # length of current flight
            self.flight_lengths = np.arange(1, 8)  # possible flight lengths
            self.flight_pl = self.flight_lengths.astype(float) ** -2
            self.flight_pl /= np.sum(self.flight_pl)  # p of each flight length

        """
        Creates a random agent for baseline comparisons. 
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
            motion: either Brownian or Levy. 
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

        if self.type == "Brownian":
            self.outputs *= 0.0
            self.outputs[:-1] += np.random.rand(len(self.outputs) - 1)

        elif self.type == "Levy":
            if (self.flight_length == 0) or (self.collision == 1):
                self.flight_length = np.random.choice(
                    a=self.flight_lengths,
                    p=self.flight_pl,
                )
                self.outputs *= 0.0
                self.outputs[:-1] += np.random.rand(len(self.outputs) - 1)

            self.flight_length -= 1


# False, False: continue
# True, False: draw new flight length
# False, True: draw new flight length
# True, True: draw new flight length
