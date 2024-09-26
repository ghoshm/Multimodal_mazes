import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import patches
from matplotlib import colors
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math


class GridPlotterContinuous:
    """
    Class responsible for visualizing the environment, agent behavior, and training progress.

    Attributes:
        agent (QLearnerAgent): The agent instance used for plotting agent-related actions and behaviors.
        pk_hw (int): The peak height or width parameter for the plotting window.
        n_steps (int): The total number of steps in the simulation or trial.
    """
    def __init__(self, agent, pk_hw, n_steps):
        self.agent = agent
        self.pk_hw = pk_hw
        self.n_steps = n_steps

    def plot_env(self, env, ax=None):
        """
        Plot the environment grid.

        Args:
        env (numpy.ndarray): The environment grid data.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, creates a new figure.

        Returns:
        ax (matplotlib.axes.Axes): The axis with the environment plot.
        """
        height, width = env[:, :, -1].shape
        if ax is None:
            fig, ax = plt.subplots(figsize=(0.1 * width, 0.1 * height))
        ax.imshow(1 - env[:, :, -1], cmap=cm.binary, alpha=0.25, zorder=1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim([self.pk_hw - 1, width - self.pk_hw])
        ax.set_ylim([height - self.pk_hw, self.pk_hw - 1])
        return ax

    def plot_prey(self, env, prey_locations, ax):
        """
        Plot prey locations on the environment grid.

        Args:
        env (numpy.ndarray): The environment grid data.
        prey_locations (list of tuples): List of prey locations to be plotted.
        ax (matplotlib.axes.Axes): The axis on which the prey is plotted.

        Returns:
        ax (matplotlib.axes.Axes): The axis with the prey plot.
        """
        cmap_ultramarine = colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:ultramarine'])
        cmap_magenta = colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:magenta'])
        combined_env = (cmap_ultramarine(env[:, :, 0]) + cmap_magenta(env[:, :, 1])) / 2
        combined_env = np.clip(combined_env, 0, 1)
        ax.imshow(combined_env, interpolation='gaussian', zorder=0)

        for prey_location in prey_locations:
            prey_patch = patches.Circle((prey_location[1], prey_location[0]), radius=0.5, color='black', zorder=2)
            ax.add_patch(prey_patch)
        return ax

    def plot_episode(self, trial_data, ax=None):
        """
        Plot the agent's path and prey locations during an episode.

        Args:
        trial_data (dict): Data of the trial containing the agent's path, prey locations, and environment.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, a new axis is created.

        Returns:
        ax (matplotlib.axes.Axes): The axis with the episode plot.
        """
        agent_path = np.array(trial_data['path'])
        prey_locations = trial_data['prey_locations'][0]
        env = trial_data['env'][0]

        if ax is None:
            ax = self.plot_env(env)
            ax = self.plot_prey(env, prey_locations, ax)

        ax.set_title(f'{len(agent_path) - 1} Steps')

        points = agent_path[:, [1, 0]]
        segments = np.concatenate([points[:-1, np.newaxis, :], points[1:, np.newaxis, :]], axis=1)

        cmap = cm.get_cmap('viridis')
        color_indices = np.linspace(0, 1, 50)
        colors = cmap(color_indices)
        segment_colors = colors[:len(segments)]
        
        ax.add_patch(patches.Rectangle((agent_path[0][1] - 0.3, agent_path[0][0] + 0.3), 0.6, 0.6, color=segment_colors[0], zorder=2))
        
        lines = LineCollection(segments, colors=segment_colors, linewidths=4, zorder=3)
        ax.add_collection(lines)
        return ax

    def plot_training_progress(self, relative_trial_lengths):
        """
        Plot the training progress showing relative episode lengths over time.

        Args:
        relative_trial_lengths (list of float): List of trial lengths relative to minimum trial lengths.

        Returns:
        plt (matplotlib.pyplot): The plot of training progress.
        """
        n_trials = len(relative_trial_lengths)
        smoothed_episode_lengths = [np.mean(relative_trial_lengths[max(0, i - 100) : i + 1]) for i in range(n_trials)]
        plt.scatter(np.arange(n_trials), relative_trial_lengths, linewidth=0, alpha=0.5, c='C0', label='Episode length')
        plt.plot(np.arange(len(smoothed_episode_lengths)), smoothed_episode_lengths, color='k', linestyle='--', linewidth=0.5, label='Smoothed')
        plt.xlabel('Trial')
        plt.ylabel('Relative Length')
        plt.legend()
        plt.title('Training Progress')
        return plt

    def plot_percentage_captured(self, training_trials):
        """
        Plot the percentage of prey captured over the training trials.

        Args:
        training_trials (list): List of trials with prey capture information.

        Returns:
        plt (matplotlib.pyplot): The plot of percentage of prey captured over trials.
        """
        n_trials = len(training_trials)
        total_prey = 0
        prey_captured = 0
        percentages = []

        for trial in training_trials:
            prey_states = trial['prey_states']
            total_prey += len(prey_states)
            prey_captured += len(prey_states) - sum(prey_states)
            percentages.append(prey_captured / total_prey)

        smoothed_percentages = [np.mean(percentages[max(0, i - 100) : i + 1]) for i in range(n_trials)]
        plt.scatter(np.arange(n_trials), percentages, linewidth=0, alpha=0.5, c='C0', label='Percentage Captured')
        plt.plot(np.arange(len(smoothed_percentages)), smoothed_percentages, color='k', linestyle='--', linewidth=0.5, label='Smoothed')
        plt.xlabel('Trial')
        plt.ylabel('Percentage Captured')
        plt.legend()
        plt.title('Training Progress')
            
    def animated_trial(self, trial_data):
        """
        Create an animated visualization of the agent's behavior during a trial.

        Args:
            trial_data (dict): Data of the trial containing the agent's path, prey locations, and environment.

        Returns:
            None: Saves the animation as a GIF.
        """
        env = trial_data['env'][0]
        ax = self.plot_env(env=env)

        cmap_ultramarine = colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:ultramarine'])
        cmap_magenta = colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:magenta'])
        
        agent_animation = ax.scatter([], [], s=120, color='k', zorder=4)
        preys_animation = [ax.scatter([], [], s=60, color='k', alpha=0.5, marker='o', zorder=3) for _ in range(len(trial_data['prey_locations'][0]))]

        walls = 1 - env[:, :, -1]
        ax.imshow(walls, cmap='binary', interpolation='nearest', alpha=0.2, zorder=1)

        height, width = env.shape[:2]
        prey_layers_image = ax.imshow(np.zeros((height, width, 4)), interpolation='gaussian', zorder=0)

        def update_animation(t):
            env_t = trial_data['env'][t]

            prey_layers = (cmap_ultramarine(env_t[:, :, 0]) + cmap_magenta(env_t[:, :, 1])) / 2
            prey_layers = np.clip(prey_layers, 0, 1)
            prey_layers_image.set_data(prey_layers)

            agent_location = trial_data['path'][t]
            agent_animation.set_offsets([agent_location[1], agent_location[0]])

            for i, prey_animation in enumerate(preys_animation):
                prey_location = trial_data['prey_locations'][t][i]
                prey_animation.set_offsets([prey_location[1], prey_location[0]])

        frames = len(trial_data['path'])
        anim = animation.FuncAnimation(ax.figure, update_animation, frames=frames, interval=200, blit=False)
        anim.save("Trial.gif", dpi=300, writer='pillow')

# Actor Network: Outputs continuous actions given a state
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Tanh()  # Assuming action range [-1, 1]
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.activation(self.output_layer(x))
        return x

# Critic Network: Estimates Q-value given state and action
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.init_weights()

    def init_weights(self):
        """Initialize network weights and biases."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, state_vector, action):
        x = torch.cat([state_vector, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output_layer(x)
        return x

# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Actor-Critic Agent using DDPG
class DDPGAgent:
    def __init__(self, input_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, tau, channels, capture_radius, location, batch_size=64):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau  # For soft update of target parameters
        self.batch_size = batch_size
        
        # Actor and Critic Networks
        self.actor = ActorNetwork(input_dim, action_dim, hidden_dim)
        self.actor_target = ActorNetwork(input_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(input_dim, action_dim, hidden_dim)
        self.critic_target = CriticNetwork(input_dim, action_dim, hidden_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer()

        # Loss function
        self.criterion = nn.MSELoss()

        self.env = None
        self.channels = channels
        self.capture_radius = capture_radius

        self.initial_location = location
        self.location = location
        self.last_action = None
        
        self.last_visble_prey_locations = None
        self.last_visible = 0

    def reset(self):
        # Reset the networks' parameters
        self.location = self.initial_location.copy()
        self.last_action = None
        self.last_visble_prey_locations = None
        self.last_visible = 0

    def weights_init(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def state_vector(self, location):
        x, y = int(location[0]), int(location[1])
        state_vector = np.zeros(self.input_dim)

        positions = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        for idx, (dx, dy) in enumerate(positions):
            nx, ny = x + dx, y + dy
            nx = min(max(nx, 0), self.env.shape[0] - 1)
            ny = min(max(ny, 0), self.env.shape[1] - 1)
            state_vector[idx] = self.env[nx, ny, 0]

        return state_vector

    def rewards(self, location, next_location, prey_location, action, next_location_valid):
        if np.linalg.norm(prey_location - location) < self.capture_radius:
            return 2000
        elif next_location_valid:
            current_distance = np.linalg.norm(location - prey_location)
            next_distance = np.linalg.norm(next_location - prey_location)
            acceleration = 0 #if self.last_action is None else self.calculate_acceleration(self.last_action[0], self.last_action[1], action[0], action[1])
            return 200 * (current_distance - next_distance + 1/current_distance) - 50 * acceleration
        else:
            return -200

    def valid_location(self, location):
        return True if self.env[int(location[0]), int(location[1]), -1] == 1 else False

    def calculate_acceleration(v1, theta1, v2, theta2, delta_t=1):
        v1x = v1 * np.cos(theta1)
        v1y = v1 * np.sin(theta1)
        
        v2x = v2 * np.cos(theta2)
        v2y = v2 * np.sin(theta2)
        
        delta_vx = v2x - v1x
        delta_vy = v2y - v1y
        
        ax = delta_vx / delta_t
        ay = delta_vy / delta_t
        
        acceleration = np.sqrt(ax**2 + ay**2)
        
        return acceleration

    def act(self, env, prey_locations, prey_speeds, prey_visible=True, noise_scale=0.0, training=False):
        self.env = env
        reward = None

        if prey_visible:
            self.last_visible_prey_locations = prey_locations
            self.last_visible = 0
        else:
            prey_locations = self.last_visible_prey_locations + prey_speeds * self.last_visible_steps
            self.last_visible += 1

        state_vector = self.state_vector(self.location)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        speed, angle = self.actor.forward(state_tensor).detach().cpu().numpy()[0]
        
        if noise_scale > 0:
            speed += np.random.normal(0, noise_scale)
            angle += np.random.normal(0, noise_scale)
            
        speed += 1
        angle *= np.pi

        next_location = self.location + np.array([speed * np.cos(angle), speed * np.sin(angle)])
        next_location_valid = self.valid_location(next_location)
        action = np.array([float(speed), float(angle)]) if next_location_valid else np.array([0, float(angle)])
        # print(f"Last action: {self.last_action}, Current action: {action}")
        
        if training:
            distances = np.linalg.norm(prey_locations - self.location, axis=1)
            nearest_prey = prey_locations[np.argmin(distances)]
            reward = self.rewards(self.location, next_location, nearest_prey, action, next_location_valid)
            next_state_vector = self.state_vector(next_location) if next_location_valid else state_vector
            done = reward >= 1000
            self.remember(state_vector, action, reward, next_state_vector, done)
            self.learn()

        if next_location_valid:
            self.location = next_location
        
        self.last_action = action.copy()

        return self.location, reward
    
    def remember(self, state_vector, action, reward, next_state_vector, done):
        self.replay_buffer.add(state_vector, action, reward, next_state_vector, done)

    def learn(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        state_vectors, actions, rewards, next_state_vectors, dones = zip(*self.replay_buffer.sample(self.batch_size))
        state_vectors = torch.FloatTensor(np.array(state_vectors))
        actions = torch.FloatTensor(np.array(actions))

        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_state_vectors = torch.FloatTensor(np.array(next_state_vectors))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Update Critic Network
        next_actions = self.actor_target.forward(next_state_vectors)
        next_q_values = self.critic_target.forward(next_state_vectors, next_actions)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        q_values = self.critic.forward(state_vectors, actions)
        critic_loss = self.criterion(q_values, target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor Network
        actor_loss = -self.critic.forward(state_vectors, self.actor(state_vectors)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def produce_plots(self, plot_training_lengths, plot_first_5_last_5, plot_percentage_captured, 
                      animate, trials, trial_lengths, pk_hw, n_steps):
        """
        Produce various plots to evaluate the agent's performance.

        Parameters:
        - trials (list): List of trial data dictionaries.
        - trial_lengths (list): List of trial lengths.
        - pk_hw: Placeholder for plotter initialization (e.g., grid size).
        - n_steps (int): Number of steps.
        - plot_training_lengths (bool): Whether to plot training progress.
        - plot_first_5_last_5 (bool): Whether to plot the first and last 5 trials.
        - plot_percentage_captured (bool): Whether to plot percentage of prey captured over trials.
        - animate (tuple): Tuple of (bool, int), where bool indicates whether to animate, and int is the trial index.
        """
        self.plotter = GridPlotterContinuous(self, pk_hw, n_steps)
        
        if plot_training_lengths:
            optimal_lengths = [self.optimal_trial_length(trials[trial]['path'][0], trials[trial]['prey_locations'][0]) for trial in trials]
            relative_trial_lengths = [trial_lengths[i] / optimal_lengths[i] for i in range(len(trial_lengths))]
            self.plotter.plot_training_progress(relative_trial_lengths=relative_trial_lengths)
        
        if plot_first_5_last_5:
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            
            for i in range(5):
                # First 5 trials
                trial_first = trials[i]
                ax_first = axs[0, i]
                self.plotter.plot_env(trial_first['env'][1], ax=ax_first)
                self.plotter.plot_prey(trial_first['env'][1], trial_first['prey_locations'][0], ax=ax_first)
                self.plotter.plot_episode(trial_first, ax=ax_first)
                
                # Last 5 trials
                trial_last = trials[len(trials)-(i+1)]
                ax_last = axs[1, 4 - i]
                self.plotter.plot_env(trial_last['env'][1], ax=ax_last)
                self.plotter.plot_prey(trial_last['env'][1], trial_last['prey_locations'][0], ax=ax_last)
                self.plotter.plot_episode(trial_last, ax=ax_last)
            plt.tight_layout()
        
        if plot_percentage_captured:
            self.plotter.plot_percentage_captured(trials)
        
        if animate[0]:
            self.plotter.animated_trial(trials[animate[1]])
    
    def optimal_trial_length(self, location, prey_locations):
        """
        Calculates the optimal path length to capture prey for static case.
        
        Parameters:
        - location (ndarray): Starting position of the agent.
        - prey_locations (ndarray): Initial locations of the prey.
        
        Returns:
        - optimal_length (int): Computed optimal length for prey capture.
        """
        current_location = np.array(location)
        remaining_prey = np.array(prey_locations).copy()
        optimal_length = 0
        
        while len(remaining_prey) > 0:
            distances = np.linalg.norm(remaining_prey - current_location, axis=1)
            idx_nearest = np.argmin(distances)
            nearest_prey = remaining_prey[idx_nearest]
            optimal_length += distances[idx_nearest]
            remaining_prey = np.delete(remaining_prey, idx_nearest, axis=0)
            current_location = nearest_prey

        return optimal_length
    
