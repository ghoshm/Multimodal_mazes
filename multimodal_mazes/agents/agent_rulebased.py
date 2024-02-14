# Rule-based agent

import numpy as np
from multimodal_mazes.agents.agent import Agent


class AgentRuleBased(Agent):
    def __init__(self, location, channels, memory_on):
        super().__init__(location, channels)
        self.type = "Rule-based"
        self.memory_on = memory_on
        self.memory = np.zeros_like(self.outputs)
        """
        Creates a rule-based agent for baseline comparisons. 
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
            memory_on: flag determining if the agent will be stateless or not [True, False].
        Properties:
            fitness: the agents mean fitness across mazes, between [0,1].
            actions: possible movements in [r,c].
            sensors: inputs in [r,c].
            channel_inputs: the agents inputs from it's location (sensors x channels).
            outputs: the policy assigned value for each action.
            memory: the summed sensory inputs from the prior time-step. 
        """

    def policy(self):
        """
        Assign a value to each action.
        Policy: move in the direction with the highest total sensory input.
            If memory_on: add the prior time steps sensory input too.
        """
        summed_inputs = np.append(np.sum(self.channel_inputs, axis=1), 0)  # x_t
        self.outputs = np.copy(summed_inputs)  # x_t

        if self.memory_on:
            self.outputs += self.memory  # x_t-1
            self.memory = np.copy(summed_inputs)  # x_t
