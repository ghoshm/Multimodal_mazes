# DQN agent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from multimodal_mazes.agents.agent import Agent
from tqdm import tqdm


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
            wm_flags:
        Properties:
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

        # Feedforward
        self.input_to_hidden = nn.Linear(
            self.n_input_units, self.n_hidden_units, bias=False
        )
        self.hidden_to_output = nn.Linear(
            self.n_hidden_units, self.n_output_units, bias=False
        )

        # Lateral
        if wm_flags[0]:
            self.input_to_input = nn.Linear(
                self.n_input_units, self.n_input_units, bias=False
            )
        if wm_flags[1]:
            self.hidden_to_hidden = nn.Linear(
                self.n_hidden_units, self.n_hidden_units, bias=False
            )
        if wm_flags[2]:
            self.output_to_output = nn.Linear(
                self.n_output_units, self.n_output_units, bias=False
            )

        # Skip
        if wm_flags[3]:
            self.input_to_output = nn.Linear(
                self.n_input_units, self.n_output_units, bias=False
            )
        if wm_flags[4]:
            self.output_to_input = nn.Linear(
                self.n_output_units, self.n_input_units, bias=False
            )

        # Feedback
        if wm_flags[5]:
            self.hidden_to_input = nn.Linear(
                self.n_hidden_units, self.n_input_units, bias=False
            )
        if wm_flags[6]:
            self.output_to_hidden = nn.Linear(
                self.n_output_units, self.n_hidden_units, bias=False
            )

    def policy(self):
        """
        Assign a value to each action.
        AgentDQN policy is a pass through a neural network.
        """
        with torch.no_grad():

            self.outputs, self.prev_input, self.hidden, self.prev_output = self.forward(
                self.prev_input, self.hidden, self.prev_output
            )

    def forward(self, prev_input, hidden, prev_output):
        """ """
        # torch.autograd.set_detect_anomaly(True)

        # Input
        new_input = torch.from_numpy(self.channel_inputs.reshape(-1)).to(torch.float32)
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
        output = torch.nn.Softmax(dim=0)(output)

        # Should add small amounts of noise to output to avoid argmax problem?

        return output, new_input, new_hidden, output

    def generate_policy(self, maze, n_steps):
        """ """
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        gamma = 0.9
        epsilons = np.repeat(
            np.linspace(start=0.95, stop=0.25, num=10), repeats=len(maze.mazes) // 10
        )

        self.gradient_norms = []

        for n in np.random.permutation(len(maze.mazes)):

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
                if torch.rand(1) < epsilons[n]:
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

            # Check for exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10)

            with torch.no_grad():
                total_norm = 0
                for p in self.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)
                self.gradient_norms.append(total_norm)

            optimizer.step()
