# Rule-based agent

import numpy as np
from matplotlib import cm
from multimodal_mazes.agents.agent import Agent


class AgentIntercept(Agent):

    policies = ['Kinetic alignment']
    colors = ['#ed45df']

    def __init__(self, location, channels, policy, direction):
        super().__init__(location, channels)

        assert policy in self.policies, "Invalid policy"
        self.type = policy
        self.channel_weights = np.copy(self.channels)
        self.time = 0
        self.direction = direction

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
            direction: the direction of kinetic alignment.
        Policies (d for direction): 
            Kinetic alignment: Move in the opposite direction of the target movement.                    
        """

    def policy(self):
        """
        Assign a value to each action based on a policy.
        """
        # Reset outputs
        self.outputs *= 0.0

        # Apply channel weights
        self.channel_inputs *= self.channel_weights

        # Limit to L and R sensors only
        # self.channel_inputs *= np.array([1, 1, 0, 0])[
        #     :, None
        # ]  

        # Implement policy            
        if self.type == "Kinetic alignment": 
            left = 1 if self.direction == -1 else 0   
            self.outputs[:] = np.append(
                np.sum(self.channel_inputs, axis=1)
                + np.sqrt(np.prod(self.channel_inputs, axis=1)),
                0,
            )
            self.outputs[0] = self.outputs[0]*left
            self.outputs[1] = self.outputs[1]*(1-left)
           
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
