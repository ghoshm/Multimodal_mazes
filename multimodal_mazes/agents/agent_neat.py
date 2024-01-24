# Evolved network agents

import numpy as np
import neat
from multimodal_mazes.agents.agent import Agent


class AgentNeat(Agent):
    def __init__(self, location, channels, sensor_noise_scale, genome, config):
        super().__init__(location, channels)
        self.type = "AgentNeat"
        self.sensor_noise_scale = sensor_noise_scale
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

        """
        Creates a NEAT agent. 
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
            sensor_noise_scale: the scale of the noise applied to every sensor. 
            genome: neat generated genome.
            config: the neat configuration holder.
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
        AgentNeat policy is a forward pass through an evolved neural network.
        """

        # Forward pass
        self.outputs = self.net.activate(list(self.channel_inputs.reshape(-1)))

        # Add noise to outputs (to avoid argmax bias)
        self.outputs += np.random.rand(len(self.outputs)) / 1000
