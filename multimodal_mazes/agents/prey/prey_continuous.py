import numpy as np

class PreyContinuous():
    def __init__(self, location, channels, scenario, motion, speed):
        """
        Creates a linear prey for linear prey tasks. 
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
            scenario: Either "Static", "Constant" or "Random".
            pm: probability of moving
            motion: "Linear".
            direction: the direction of prey movement.
        Properties:
        """    
        self.location = np.array(location)
        self.channels = np.array(channels)
        self.scenario = scenario
        self.motion = motion
        self.speed = speed
        self.collision = False
        
    def move(self, env):
        """
        Updates the agent's state by one action.
        Arguments:
            env: a np array of size x size x channels + 1.
                Where [:,:,-1] stores the environment structure.
        Updates:
            self.location: by one action.
            If the action collides with a wall it is ignored.
        """

        if self.scenario != "Static":
              
            if env[int(self.location[0]), int(self.location[1] + 0.5 + self.speed), -1] == 1.0:
                self.location += self.speed
                self.collision = 0

            else:
                self.collision = 1