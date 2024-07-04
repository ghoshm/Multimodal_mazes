# Linear prey

import numpy as np

class PreyLinear():
    def __init__(self, location, channels, motion, direction):
        
        self.location = np.array(location)
        self.channels = np.array(channels)
        self.type = motion
        self.collision = 0
        self.direction = direction
        
        """
        Creates a linear prey for linear prey tasks. 
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
            motion: either linear or ...
            direction: the direction the prey moves
        Properties:
            
        """                
        
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
        if self.type == 'Linear':
            
            if ( 
                env[
                    self.location[0],
                    self.location[1]+self.direction,
                    -1,
                ] 
                == 1.0
            ):
                self.location += [0, self.direction]
                self.collision = 0

            else:
                self.collision = 1