# DQN agent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multimodal_mazes
from multimodal_mazes.agents.agent import Agent
from tqdm import tqdm

# See below for a draft of more flexible version


class AgentDQN(nn.Module, Agent):
    def __init__(
        self, location, channels, sensor_noise_scale, n_hidden_units, wm_flags
    ):
        """
        Creates a DQN agent.
        Arguments:
            location: initial position [r,c].
            channels: list of active (1) and inative (0) channels e.g. [0,1].
            sensor_noise_scale: the scale of the noise applied to every sensor.
            n_hidden_units: the number of units in the hidden layer.
            wm_flags: a 7 element binary vector, which includes or excludes each additional weight matrix.
        """

        # Set up
        assert wm_flags.shape == (7,), "wm_flags must have 7 elements"
        nn.Module.__init__(self)
        Agent.__init__(self, location, channels)
        self.type = "AgentDQN"
        self.sensor_noise_scale = sensor_noise_scale
        self.wm_flags = wm_flags

        # Units
        self.n_input_units = len(self.channel_inputs.reshape(-1))
        self.n_hidden_units = n_hidden_units
        self.n_output_units = len(self.outputs)

        # Number of connections to units in each layer
        in_features = [0, self.n_input_units, self.n_hidden_units]

        if wm_flags[0]:  # ii
            in_features[0] += self.n_input_units

        if wm_flags[1]:  # hh
            in_features[1] += self.n_hidden_units

        if wm_flags[2]:  ## oo
            in_features[2] += self.n_output_units

        if wm_flags[3]:  # io
            in_features[2] += self.n_input_units

        if wm_flags[4]:  # oi
            in_features[0] += self.n_output_units

        if wm_flags[5]:  # hi
            in_features[0] += self.n_hidden_units

        if wm_flags[6]:  # oh
            in_features[1] += self.n_output_units

        self.ks = np.sqrt(1 / np.array(in_features))

        # Feedforward
        self.input_to_hidden = nn.Linear(
            self.n_input_units, self.n_hidden_units, bias=False
        )
        nn.init.uniform_(self.input_to_hidden.weight, a=-self.ks[1], b=self.ks[1])

        self.hidden_to_output = nn.Linear(
            self.n_hidden_units, self.n_output_units, bias=False
        )
        nn.init.uniform_(self.hidden_to_output.weight, a=-self.ks[2], b=self.ks[2])

        # Lateral
        if wm_flags[0]:  # ii
            self.input_to_input = nn.Linear(
                self.n_input_units, self.n_input_units, bias=False
            )
            nn.init.uniform_(self.input_to_input.weight, a=-self.ks[0], b=self.ks[0])

        if wm_flags[1]:  # hh
            self.hidden_to_hidden = nn.Linear(
                self.n_hidden_units, self.n_hidden_units, bias=False
            )
            nn.init.uniform_(self.hidden_to_hidden.weight, a=-self.ks[1], b=self.ks[1])

        if wm_flags[2]:  # oo
            self.output_to_output = nn.Linear(
                self.n_output_units, self.n_output_units, bias=False
            )
            nn.init.uniform_(self.output_to_output.weight, a=-self.ks[2], b=self.ks[2])

        # Skip
        if wm_flags[3]:  # io
            self.input_to_output = nn.Linear(
                self.n_input_units, self.n_output_units, bias=False
            )
            nn.init.uniform_(self.input_to_output.weight, a=-self.ks[2], b=self.ks[2])

        if wm_flags[4]:  # oi
            self.output_to_input = nn.Linear(
                self.n_output_units, self.n_input_units, bias=False
            )
            nn.init.uniform_(self.output_to_input.weight, a=-self.ks[0], b=self.ks[0])

        # Feedback
        if wm_flags[5]:  # hi
            self.hidden_to_input = nn.Linear(
                self.n_hidden_units, self.n_input_units, bias=False
            )
            nn.init.uniform_(self.hidden_to_input.weight, a=-self.ks[0], b=self.ks[0])

        if wm_flags[6]:  # oh
            self.output_to_hidden = nn.Linear(
                self.n_output_units, self.n_hidden_units, bias=False
            )
            nn.init.uniform_(self.output_to_hidden.weight, a=-self.ks[1], b=self.ks[1])

    def policy(self):
        """
        Assign a value to each action.
        AgentDQN policy is a pass through a neural network.
        """
        with torch.no_grad():

            self.outputs, self.prev_input, self.hidden, self.prev_output = self.forward(
                self.prev_input, self.hidden, self.prev_output
            )

    def forward(self, prev_input, hidden, prev_output, tensor_input=False):
        """
        Performs a forward pass through a DQN model.
            Note: input activations at t, come from self.
        Arguments:
            prev_input: input activations at t-1.
            hidden: hidden activations at t-1.
            prev_output: output activations at t-1.
            tensor_input: set to true if you need to input tensors,
                instead of numpy arrays.
        Returns:
            output: output activations at t [used as q-values].
            new_input: input activations at t.
            new_hidden: hidden activations at t.
            output: output activations at t.
        """
        # torch.autograd.set_detect_anomaly(True)

        # Input
        if tensor_input == False:
            new_input = torch.from_numpy(self.channel_inputs.reshape(-1)).to(
                torch.float32
            )
        elif tensor_input == True:
            new_input = self.channel_inputs

        if self.wm_flags[0]:  # Lateral
            new_input = new_input + self.input_to_input(prev_input)
        if self.wm_flags[4]:  # Skip
            new_input = new_input + self.output_to_input(prev_output)
        if self.wm_flags[5]:  # Feedback
            new_input = new_input + self.hidden_to_input(hidden)

        # Hidden
        new_hidden = self.input_to_hidden(new_input)
        if self.wm_flags[1]:  # Lateral
            new_hidden = new_hidden + self.hidden_to_hidden(hidden)
        if self.wm_flags[6]:  # Feedback
            new_hidden = new_hidden + self.output_to_hidden(prev_output)
        new_hidden = torch.relu(new_hidden)

        # Output
        output = self.hidden_to_output(new_hidden)
        if self.wm_flags[2]:  # Lateral
            output = output + self.output_to_output(prev_output)
        if self.wm_flags[3]:  # Skip
            output = output + self.input_to_output(new_input)
        output = output + torch.rand(len(output)) / 1000
        output = torch.nn.Softmax(dim=0)(output)

        return output, new_input, new_hidden, output

    def generate_policy(self, maze, n_steps, maze_test=None):
        """
        Uses deep Q-learning to optimise model weights.
        Arguments:
            maze: a class containing a set of mazes.
            n_steps: number of simulation steps.
            maze_test: a class containing a set of mazes.
                Used to record the agent's fitness 100 times throughout training.
        Updates:
            self.parameters.
            self.training_fitness (if maze_test is provided).
        """
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        gamma = 0.9
        epsilons = np.repeat(
            np.linspace(start=0.95, stop=0.25, num=10), repeats=len(maze.mazes) // 10
        )

        self.gradient_norms = []
        self.training_fitness = []

        for a, n in enumerate(np.random.permutation(len(maze.mazes))):

            # Record fitness
            if (a % (len(maze.mazes) // 100) == 0) & (maze_test != None):
                with torch.no_grad():
                    fitness = multimodal_mazes.eval_fitness(
                        genome=None,
                        config=None,
                        channels=self.channels,
                        sensor_noise_scale=self.sensor_noise_scale,
                        drop_connect_p=0.0,
                        maze=maze_test,
                        n_steps=n_steps,
                        agnt=self,
                    )
                    self.training_fitness.append(fitness)

            # Reset agent
            prev_input = torch.zeros(self.n_input_units)
            hidden = torch.zeros(self.n_hidden_units)
            prev_output = torch.zeros(self.n_output_units)

            self.location = np.copy(maze.start_locations[n])
            self.outputs = torch.zeros(self.n_output_units)

            loss = 0.0

            # Starting reward
            starting_reward = (
                maze.d_maps[n].max()
                - maze.d_maps[n][self.location[0], self.location[1]]
            ) / maze.d_maps[n].max()

            # Trial
            for time in range(n_steps):
                # Sense
                self.sense(maze.mazes[n])

                # Epsilon-greedy action selection
                if torch.rand(1) < epsilons[a]:
                    action = torch.randint(
                        low=0, high=self.n_output_units, size=(1,)
                    ).item()
                else:
                    with torch.no_grad():
                        q_values, _, _, _ = self.forward(
                            prev_input, hidden, prev_output
                        )
                        action = torch.argmax(q_values).item()

                # Predicted Q-value
                q_values, prev_input, hidden, prev_output = self.forward(
                    prev_input.detach(), hidden.detach(), prev_output.detach()
                )
                predicted = q_values[action]

                # Act
                self.outputs *= 0.0
                self.outputs[action] = 1.0
                self.act(maze.mazes[n])

                # Reward
                reward = (
                    maze.d_maps[n].max()
                    - maze.d_maps[n][self.location[0], self.location[1]]
                ) / maze.d_maps[n].max()
                reward -= starting_reward
                reward *= 2

                # Target Q-value
                self.sense(maze.mazes[n])
                with torch.no_grad():
                    next_q_values, _, _, _ = self.forward(
                        prev_input, hidden, prev_output
                    )
                    target = reward + (gamma * torch.max(next_q_values)) - 0.1

                # Loss
                loss = loss + criterion(predicted, target)

                if np.array_equal(self.location, maze.goal_locations[n]):
                    break

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)

            # Check for exploding gradients
            # with torch.no_grad():
            #     total_norm = 0
            #     for p in self.parameters():
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            #     total_norm = total_norm ** (1.0 / 2)
            #     self.gradient_norms.append(total_norm)

            optimizer.step()


# Idea for an agent with a more flexible architecture
# layers = {'input': nb_input, 'hidden': nb_hidden, 'output': nb_output}
# connection_keys = {('input', 'hidden'), ('hidden', 'output')}
# self.connections = dict()
# # sanity check:
# for c_in, c_out in connection_keys:
#   self.connections[c_in, c_out] = x = nn.Linear(layers[c_in], layers[c_out])
#   setattr(self, c_in+'_to_'+c_out, x)


# all_connection_types = [(c_in, c_out) for c_in in layers.keys() for c_out in layers.keys()]
# # itertools.combinations might not be the right function! iterate over all subsets is the idea
# all_specifications = [spec for spec in itertools.combinations(all_connection_types) if validate(spec)]

# def validate(connection_type, layers):
#   if ('input', 'hidden') in connection_type and ('hidden', 'output') in connection_type:
#     return True
#   else:
#     return False

# def validate(connection_type, layers):
#   pass # check if there is a path from 'input' to 'output'
#   # check this, but also check that every layer is used
