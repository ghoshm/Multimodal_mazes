# Rule-based agent

import numpy as np
from matplotlib import cm
from multimodal_mazes.agents.agent import Agent


class AgentRuleBased(Agent):

    policies = [
        "-/-",  # random baseline
        "+/-",  # unimodal
        "-/+",  # unimodal
        "2l-binary",  # 2-look
        "2l-max",  # 2-look
        "Max-dv",  # max-dv
        "Linear fusion",  # multimodal
        "Nonlinear fusion",  # multimodal
    ]

    colors = cm.get_cmap("plasma", len(policies)).colors.tolist()

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

        self.channel_weights = np.copy(self.channels)

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

        # Apply channel weights
        self.channel_inputs *= self.channel_weights

        # self.channel_inputs *= np.array([1, 1, 0, 0])[
        #     :, None
        # ]  # uncomment to limit solutions to only L and R sensors.

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

    def policy_wrapper(self, channel_inputs):
        """
        Returns an action output for each channel input in a batch.
        Arguments:
            channel_inputs: inputs from locations (sensors x channels x batch).
        Returns:
            output: a np vector of actions.
        """
        output = []
        for n in range(channel_inputs.shape[2]):
            self.channel_inputs = np.copy(channel_inputs[:, :, n])  # "sense"
            self.policy()  # policy
            output.append(np.argmax(self.outputs))  # action
        output = np.array(output)

        return output

    def fit_channel_weights(self, n_weights, channel_inputs, ci_actions):
        """
        Find the best weight per channel via a grid search.
        Arguments:
            n_weights: the number of weights to test per channel.
            channel_inputs: inputs from locations (sensors x channels x batch).
            ci_actions: the correct action for each channel_input (np vector).
        Updates:
            channel_weights: the weight to apply to each channel's sensory inputs.
        Note:
            Assumes two channels.
            Will skip one-look algorithms (-/-, +/-, -/+).
        """

        if sum(self.channels) >= 2:
            weights = np.linspace(start=0.0, stop=1.0, num=n_weights)
            results = np.zeros((n_weights, n_weights))

            # Test weight combinations
            for a, ch0 in enumerate(weights):
                for b, ch1 in enumerate(weights):
                    self.channel_weights = [ch0, ch1]
                    outputs = self.policy_wrapper(channel_inputs)
                    results[a, b] = sum(outputs == ci_actions) / len(outputs)

            # Choose best weights
            r, c = np.where(results == results.max())
            channel_weights = np.array([weights[r[-1]], weights[c[-1]]])
            channel_weights /= sum(channel_weights)
            self.channel_weights = np.copy(channel_weights)
