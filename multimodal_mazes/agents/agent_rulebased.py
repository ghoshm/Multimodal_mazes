# Rule-based agent

import numpy as np
from multimodal_mazes.agents.agent import Agent


class AgentRuleBased(Agent):

    policies = [
        "-/-",  # random baseline
        "+/-",  # unimodal
        "-/+",  # unimodal
        "Max-dv",  # max-dv
        "2l-binary",  # 2-look
        "2l-max",  # 2-look
        "Linear fusion",  # multimodal
        "Nonlinear fusion",  # multimodal
    ]

    def __init__(self, location, channels, policy):
        super().__init__(location, channels)

        assert policy in self.policies, "Invalid policy"
        self.type = policy
        if self.type == "-/-":
            self.channels = [0, 0]
        elif self.type == "+/-":
            self.channels = [1, 0]
        elif self.type == "-/+":
            self.channels = [0, 1]

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
        Policies (d for direction): 
            -/-: move randomly. 
            +/-: move in the d with the most input from Ch0. 
            -/+: move in the d with the most input from Ch1. 
            Max-dv: 
            2-look algorithms: 
                choose the d with the most input per channel 
                if Ch0-d == Ch1-d:
                    move in d 
                else: 
                    Binary:
                        randomly choose from Ch0-d and Ch1-d'
                    Max: 
                        choose Ch-d with the most input.        
            Linear fusion: move in the d with the most (Ch0 + Ch1) input. 
            Nonlinear fusion: linear fusion + a nonlinear term.  
        """

    def policy(self):
        """
        Assign a value to each action based on a policy.
        """
        # Reset outputs
        self.outputs *= 0.0

        # Implement policy
        if self.type == "Max-dv":
            self.outputs[:] = np.append(np.max(self.channel_inputs, axis=1), 0)

        elif self.type == "2l-binary":
            self.channel_inputs += np.random.rand(*self.channel_inputs.shape) / 1000
            self.outputs[np.argmax(self.channel_inputs, axis=0)] = 1

        elif self.type == "2l-max":
            self.channel_inputs += np.random.rand(*self.channel_inputs.shape) / 1000
            self.outputs[np.argmax(self.channel_inputs, axis=0)] += np.max(
                self.channel_inputs, axis=0
            )

        elif self.type == "Nonlinear fusion":
            self.outputs[:] = np.append(
                np.sum(self.channel_inputs, axis=1)
                + np.sqrt(np.prod(self.channel_inputs, axis=1)),
                0,
            )

        else:
            "Linear policies: -/-, +/-, -/+, Linear fusion"
            self.outputs[:] = np.append(np.sum(self.channel_inputs, axis=1), 0)

        # Add noise to outputs (to avoid argmax bias)
        self.outputs += np.random.rand(len(self.outputs)) / 1000
