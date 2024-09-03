import numpy as np

# 4 hidden units: up, down, left, right
# 4 (or 5) outputs: up, down, left, right, (pause)
# 

class AgentCombination():

    def __init__(
            self, 
            location,
            channels,
            skip, # Skip connection from input to output
            lateral_input, # Lateral connection between inputs
            lateral_hidden, # Lateral connection between hidden layer
            lateral__output, # Lateral connection between outputs
            feedback_hidden_input, # 
            feedback_output_hidden,
    ):
        self.location = location
        self.channels = channels
        self.skip = skip
        self.lateral_input = lateral_input
        self.lateral_hidden = lateral_hidden
        self.lateral_output = lateral__output
        self.feedback_hidden_input = feedback_hidden_input
        self.feedback_output_hidden = feedback_output_hidden
