import numpy as np

class PreyLinear():
    def __init__(self, location, channels, scenario, motion, direction, pm):
        """
        Creates a linear prey for linear prey tasks. 
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
            scenario: Either "Static", "Constant" or "Random".
            pm: probability of moving
            motion: "Linear", "Disappearing", "Brownian" or "Levy".
            direction: the direction of prey movement.
        Properties:
        """    
        self.location = np.array(location)
        self.channels = np.array(channels)
        self.type = scenario
        self.collision = 0
        self.direction = direction
        self.pm = pm
        self.motion = motion

        if self.motion == "Levy":
            self.flight_length = 0  # length of current flight
            self.flight_lengths = np.arange(1, 8)  # possible flight lengths
            self.flight_pl = self.flight_lengths.astype(float) ** -2
            self.flight_pl /= np.sum(self.flight_pl)  # p of each flight length            
        
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

        # Act (if the action does not collide with a wall)
        if self.type != "Static":
              
            if self.type == "Random":
                # Change direction for random motion
                possible_directions = [-1, 1]
                self.direction = possible_directions[np.random.choice(range(2))]

                if self.motion == "Levy" and self.flight_length == 0:
                    self.flight_length = np.random.choice(
                        a=self.flight_lengths,
                        p=self.flight_pl,
                    )
            
 
            if env[self.location[0], self.location[1] + self.direction, -1] == 1.0:
                self.location += np.array([0, self.direction])
                self.collision = 0

                if self.motion == "Levy":
                    self.flight_length -= 1

            else:
                self.collision = 1