# Rule-based agent

import numpy as np
from multimodal_mazes.agents.agent import Agent


class AgentRuleBasedMemory(Agent):

    policies = ["Recurrent outputs", "Hidden skip"]

    def __init__(self, location, channels, policy):
        super().__init__(location, channels)

        assert policy in self.policies, "Invalid policy"
        self.type = policy

        self.alpha = 1.0

        if self.type == "Hidden skip":
            self.memory = np.zeros_like(self.outputs)

        """
        Creates a rule-based agent which follows a specific policy. 
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
        Properties:
            fitness: the agents mean fitness across mazes, between [0,1].
            actions: possible movements in [r,c].
            sensors: inputs in [r,c].
            channel_inputs: the agents inputs from it's location (sensors x channels).
            outputs: the policy assigned value for each action.
            alpha: the weight applied to the memory. 
        Policies (d for direction): 
            Recurrent outputs: linear fusion (t) + weighted ouputs (t-1).  
            Hidden skip: linear fusion (t) + weighted inputs (t-1). 
        """

    def policy(self):
        """
        Assign a value to each action based on a policy.
        """

        # self.channel_inputs *= np.array([1, 1, 0, 0])[:, None] # uncomment to limit solutions to only L and R sensors.

        # Implement policy
        if self.type == "Recurrent outputs":

            self.outputs[:] = np.append(np.sum(self.channel_inputs, axis=1), 0) + (
                self.alpha * self.outputs
            )

        elif self.type == "Hidden skip":

            self.outputs[:] = np.append(np.sum(self.channel_inputs, axis=1), 0) + (
                self.alpha * self.memory
            )

            self.memory[:] = np.append(np.sum(self.channel_inputs, axis=1), 0)

        # Add noise to outputs (to avoid argmax bias)
        self.outputs += np.random.rand(len(self.outputs)) / 1000
