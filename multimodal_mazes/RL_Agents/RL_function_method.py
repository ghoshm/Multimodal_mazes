import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import patches
from matplotlib import colors
import matplotlib.animation as animation

class GridPlotter:
    """
    Class responsible for visualizing the environment, agent behavior, and training progress.

    Attributes:
    agent (QLearnerAgent): The agent instance used for plotting agent-related actions and behaviors.
    """
    def __init__(self, agent):
        self.agent = agent

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
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim([self.agent.pk_hw - 1, width - self.agent.pk_hw])
        ax.set_ylim([height - self.agent.pk_hw, self.agent.pk_hw - 1])
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
        for prey_location in prey_locations:
            prey_patch = patches.Circle(xy=(prey_location[1], prey_location[0]), radius=0.5, color='black', zorder=2)
            ax.add_patch(prey_patch)
            ax.imshow((colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:ultramarine'])(env[:, :, 0]) + colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:magenta'])(env[:, :, 1]))/2, interpolation='gaussian', zorder=0) 
        return ax

    def plot_agent(self, ax, agent_location, color):
        """
        Plot the agent's position on the environment grid.

        Args:
        ax (matplotlib.axes.Axes): The axis on which the agent will be plotted.
        agent_location (tuple): The (x, y) coordinates of the agent.
        color (str or tuple): The color of the agent.

        Returns:
        ax (matplotlib.axes.Axes): The axis with the agent plot.
        """
        agent_patch = patches.Rectangle((agent_location[1] - 0.5, agent_location[0] - 0.5), 0.8, 0.8, color=color, zorder=3)
        ax.add_patch(agent_patch)
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
        agent_path = trial_data['path']
        prey_locations = trial_data['prey_locations'][0]
        env = trial_data['env'][0]

        if ax is None:
            ax = self.plot_env(env)
            ax = self.plot_prey(env, prey_locations, ax)

        ax.set_title(f'{len(agent_path)} Steps')

        for i in range(len(agent_path)):
            trial_frac = i / self.agent.n_steps
            color=cm.get_cmap('viridis')(trial_frac)
            ax = self.plot_agent(ax, agent_location=agent_path[i], color = color)
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

        for trial in range(n_trials):
            prey_states = training_trials[trial]['prey_states']
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
        ax = self.plot_env(env=trial_data['env'][1])
        agnt_animation = ax.scatter([], [], s=120, color='k', zorder=3)
        preys_animation = [ax.scatter([], [], s=60, color='k', alpha=0.5, marker='o', zorder=2) for _ in range(len(trial_data['prey_locations'][0]))]

        def update_animation(t):
            env = trial_data['env']
            combined_env = (colors.LinearSegmentedColormap.from_list("", ["white", "xkcd:ultramarine"])(env[t][:,:,0]) + colors.LinearSegmentedColormap.from_list("", ["white", "xkcd:magenta"])(env[t][:,:,1])) / 2
            combined_env = np.clip(combined_env, 0, 1)  # Ensure the values are between 0 and 1
            ax.imshow(combined_env, interpolation='gaussian', zorder=0)
            
            agnt_animation.set_offsets(list(trial_data['path'][t])[::-1])
            for i in range(len(preys_animation)):
                preys_animation[i].set_offsets(list(trial_data['prey_locations'][t][i])[::-1])
    
        anim = animation.FuncAnimation(ax.figure, update_animation, frames=range(0, len(trial_data['path'][:30])), blit=False)
        anim.save("Trial.gif", dpi=300)     


class QLearnerAgent:
    """
    The QLearnerAgent class implements a reinforcement learning agent using Q-learning.
    The agent navigates through an environment, senses features, and updates its policy based
    on the rewards received during training.
    
    Attributes:
        pk_hw: Agent's perception kernel size (height/width).
        channels: Number of channels in the environment (e.g., sensory inputs).
        actions: A dictionary of possible actions the agent can take.
        reset_location: The initial position of the agent.
        sensor_noise_scale: The noise scale applied to the agent's sensors.
        n_steps: Number of steps in a trial.
        n_features: Number of features the agent can sense.
        cost_per_step: Cost incurred per step taken by the agent.
        cost_per_collision: Cost incurred for collisions with obstacles.
        alpha: Learning rate for Q-learning updates.
        epsilon: Exploration rate for epsilon-greedy policy.
        gamma: Discount factor for future rewards.
    """
    def __init__(self, pk_hw, channels, actions, location, sensor_noise_scale, n_steps, n_features, cost_per_step, cost_per_collision, alpha, epsilon, gamma):
        self.pk_hw = pk_hw
        self.channels = np.array(channels)
        self.actions = actions
        self.reset_location = np.array(location)
        self.location = np.array(location)
        self.sensor_noise_scale = sensor_noise_scale
        self.n_steps = n_steps
        
        self.n_features = n_features
        self.n_actions = len(actions)
        self.cost_per_step = cost_per_step
        self.cost_per_collision = cost_per_collision
        self.max_distance = 21
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.theta = np.zeros((n_features, self.n_actions))
        self.last_action = None
        
    def reset(self):
        """Resets the agent to its initial location and direction."""
        self.location = np.copy(self.reset_location)
        self.agent_direction = 0
        
    def sense_features(self, location):
        """
        Extracts sensory features for the agent based on its current location.
        
        Args:
            location: Current location of the agent.
        
        Returns:
            features: A NumPy array containing sensory information.
        """
        features = np.zeros((self.n_features))
        
        features[0] = self.env[location[0] - 1, location[1], 0]         # Sensory information up
        features[1] = self.env[location[0] + 1, location[1], 0]         # Sensory information down
        features[2] = self.env[location[0], location[1] + 1, 0]         # Sensory information right
        features[3] = self.env[location[0], location[1] - 1, 0]         # Sensory information left
        features[4] = self.env[location[0] - 1, location[1] + 1, 0]     # Sensory information up-right
        features[5] = self.env[location[0] - 1, location[1] - 1, 0]     # Sensory information up-left
        features[6] = self.env[location[0] + 1, location[1] + 1, 0]     # Sensory information down-right
        features[7] = self.env[location[0] + 1, location[1] - 1, 0]     # Sensory information down-left
        
        prey_index, scaled_distance, scaled_angle = self.closest_prey_features(self.prey_locations, location)
        
        features[8] = scaled_distance                                   # Current distance
        features[9] = scaled_angle                                      # Current angle
        
        offsets = np.zeros_like(self.prey_locations)
        offsets[:, 1] = self.prey_directions * self.last_seen if np.random.rand() > self.prey_pm else offsets[:, 1]
        predicted_next_prey_locations = self.prey_locations + offsets
        _, pred_scaled_distance, pred_scaled_angle = self.closest_prey_features(predicted_next_prey_locations, location)

        features[10] = pred_scaled_distance     # Predicted distance
        features[11] = pred_scaled_angle        # Predicted angle

        return features

    def closest_prey_features(self, prey_locations, location):
        """
        Finds the closest prey to the agent and calculates the distance and angle to the prey.
        
        Args:
            prey_locations: List of prey locations.
            location: Current location of the agent.
        
        Returns:
            nearest_prey_index: Index of the nearest prey.
            scaled_distance: Scaled distance to the nearest prey.
            scaled_angle: Scaled angle to the nearest prey.
        """
        nearest_prey_index, nearest_prey = min(enumerate(prey_locations), key=lambda prey: np.linalg.norm(location - np.array(prey[1])))
        delta = location - nearest_prey[0]
        # scaled_distance = np.linalg.norm(delta) / self.max_distance
        scaled_distance = (abs(delta[0]) + abs(delta[1])) / self.max_distance
        angle = np.arctan2(delta[1], delta[0])
        # scaled_angle = angle / (2 * np.pi) if angle >= 0 else (angle + 2 * np.pi) / (2 * np.pi)
        scaled_angle = angle / np.pi
        return nearest_prey_index, scaled_distance, scaled_angle
        
    def policy(self, action):
        """
        Defines the agent's policy to determine the next location and reward based on the action.
        
        Args:
            action: The chosen action.
        
        Returns:
            next_location: The agent's next location.
            reward: The reward obtained after taking the action.
        """
        next_location = self.location + self.actions[action]['delta']
        next_distance = self.closest_prey_features(self.prey_locations, next_location)[1]
        closest_distance_difference = next_distance - self.closest_prey_features(self.prey_locations, self.location)[1]
        self.last_action = action if self.last_action is None else self.last_action
        
        if self.env[next_location[0], next_location[1], -1] == 1:
            if next_location in self.prey_locations:
                reward = 1000
            elif closest_distance_difference < 0:
                reward = self.cost_per_step + (self.max_distance * 10 / (next_distance * self.max_distance))
            else:
                reward = self.cost_per_step #- (self.max_distance * 10 / (next_distance * self.max_distance))

            if action != self.last_action:
                delta = self.actions[action]['delta']
                reward -= (200 * (abs(delta[0]) + abs(delta[1])))

        else:
            next_location = np.copy(self.location)
            reward = self.cost_per_collision

        return next_location, reward
    
    def act(self, env, prey_locations, prey_directions, prey_pm, visible=True):
        """
        Executes the agent's action using the epsilon-greedy policy during evaluation.
        
        Args:
            env: The current environment state.
            prey_locations: The current prey locations.
            prey_directions: Directions of the prey.
            prey_pm: Motion parameter for prey (randomness in movement).
            visible: If the prey is visible or not (default is True).
        
        Returns:
            next_location: The next location of the agent.
        """
        self.env = env
        self.prey_pm = prey_pm
        self.prey_directions = prey_directions
        
        if visible:
            self.last_seen_prey_locations = prey_locations
            self.prey_locations = self.last_seen_prey_locations
            self.last_seen = 0
        else:
            self.last_seen += 1
            offsets = np.zeros_like(self.last_seen_prey_locations)
            offsets[:, 1] = (prey_directions * self.last_seen * self.prey_pm).astype(int)
            self.prey_locations = self.last_seen_prey_locations + offsets

        action = self.epsilon_greedy_policy(0, self.location)      
        next_location, _ = self.policy(action)
        self.location = next_location
        return next_location
        
    def training_act(self, env, prey_locations, prey_directions, prey_pm, visible=True):
        """
        Executes the agent's action during training and updates the Q-values.
        
        Args:
            env: The current environment state.
            prey_locations: The current prey locations.
            prey_directions: Directions of the prey.
            prey_pm: Motion parameter for prey (randomness in movement).
            visible: If the prey is visible or not (default is True).
        
        Returns:
            next_location: The next location of the agent.
            reward: The reward received from the action.
        """
        self.env = env
        self.prey_pm = prey_pm
        self.prey_directions = prey_directions
        
        if visible:
            self.last_seen_prey_locations = prey_locations
            self.prey_locations = self.last_seen_prey_locations
            self.last_seen = 0
        else:
            self.last_seen += 1
            offsets = np.zeros_like(self.last_seen_prey_locations)
            offsets[:, 1] = (prey_directions * self.last_seen * self.prey_pm).astype(int)
            self.prey_locations = self.last_seen_prey_locations + offsets
        
        action = self.epsilon_greedy_policy(self.epsilon, self.location)        
        next_location, reward = self.policy(action)
        next_action = self.epsilon_greedy_policy(0, next_location)
        
        self.learn(self.location, next_location, action, next_action, reward, self.alpha)
        self.location = next_location

        self.alpha = self.update_parameter(self.alpha, 0.99, 0.05, reward)
        self.epsilon = self.update_parameter(self.epsilon, 0.99, 0.01, reward)

        return next_location, reward
        
    def learn(self, location, next_location, action, next_action, reward, alpha):
        """
        Updates the Q-values using temporal difference (TD) learning.
        
        Args:
            location: The current location of the agent.
            next_location: The agent's location after taking the action.
            action: The action taken by the agent.
            next_action: The next action chosen by the agent.
            reward: The reward received from the action.
            alpha: The learning rate.
        
        Returns:
            TD_error: The temporal difference error.
        """
        Q = float(self.Q_value(location)[action])
        Q_next =  float(self.Q_value(next_location)[next_action])
        TD_error = float(reward) + (float(self.gamma) * float(Q_next)) - float(Q)
        self.theta[:, action] += float(alpha) * float(TD_error) * self.sense_features(location)            
        return TD_error

    def Q_value(self, location):
        """
        Calculates the Q-values for the agent at the given location.
        
        Args:
            location: The current location of the agent.
        
        Returns:
            Q_values: A NumPy array containing Q-values for all actions.
        """
        return np.dot(self.sense_features(location), self.theta)

    def epsilon_greedy_policy(self, epsilon, location):
        """
        Chooses an action based on the epsilon-greedy policy.
        
        Args:
            epsilon: The exploration rate.
            location: The current location of the agent.
        
        Returns:
            action: The selected action.
        """
        return np.random.randint(self.n_actions) if (np.random.rand() < epsilon) else np.argmax(self.Q_value(location))
    
    def update_parameter(self, parameter, decay_rate, parameter_min, reward=None):
        """
        Updates a parameter (e.g., learning rate, exploration rate) by decaying it.
        
        Args:
            parameter: The parameter to be updated.
            decay_rate: The rate at which the parameter decays.
            parameter_min: The minimum allowable value for the parameter.
            reward: The reward received (optional).
        
        Returns:
            Updated parameter value.
        """
        return max(parameter * decay_rate, parameter_min)
    
    def minimum_trial_length(self, location, prey_locations):
        """
        Calculates the minimum number of steps required to capture all prey.
        
        Args:
            location: The current location of the agent.
            prey_locations: A list of prey locations.
        
        Returns:
            min_length: The minimum number of steps to capture all prey.
        """
        min_length = 0
        while len(prey_locations) > 0:
            nearest_prey = min(prey_locations, key=lambda reward: np.linalg.norm(location - np.array(reward)))
            delta = location - prey_locations[0]
            min_length += abs(delta[0]) + abs(delta[1])
            location = nearest_prey        
            prey_locations = np.delete(nearest_prey, np.argwhere(prey_locations == nearest_prey))
        return min_length
    
    def produce_plots(self, training_lengths, first_5_last_5, percentage_captured, animate, trials, trial_lengths):
        """
        Generates plots and visualizations of the agent's training progress, including
        training length plots, visual comparisons of early vs late episodes, and animations.
        
        Args:
            training_lengths: Whether to plot training progress based on trial lengths.
            first_5_last_5: Whether to plot the first 5 and last 5 episodes.
            percentage_captured: Whether to plot the percentage of prey captured over trials.
            animate: Whether to create an animated trial (tuple of boolean and trial index).
            trials: A list of trial data.
            trial_lengths: A list of trial lengths.
        
        Returns:
            None: Displays the plots and saves the animation (if enabled).
        """
        self.plotter = GridPlotter(self)
        if training_lengths:
            minimum_trial_lengths = [self.minimum_trial_length(trials[i]['path'][0], trials[i]['prey_locations'][0]) for i in range(len(trial_lengths))]
            relative_trial_lengths = [trial_lengths[i] / minimum_trial_lengths[i] for i in range(len(trial_lengths))]
            self.plotter.plot_training_progress(relative_trial_lengths=relative_trial_lengths)
        if first_5_last_5:
            fig, axs = plt.subplots(2, 5, figsize=(10, 4))
            for i in range(5):
                axs[0, i] = self.plotter.plot_env(trials[i]['env'][1], ax=axs[0, i])
                axs[1, (4 - i)] = self.plotter.plot_env(trials[len(trials) - i - 1]['env'][1], ax=axs[1, (4 - i)])
                axs[0, i] = self.plotter.plot_prey(trials[i]['env'][1], trials[i]['prey_locations'][0], ax=axs[0, i])
                axs[1, (4 - i)] = self.plotter.plot_prey(trials[len(trials) - i - 1]['env'][1], trials[len(trials) - i - 1]['prey_locations'][0], ax=axs[1, (4 - i)])
                
                self.plotter.plot_episode(trials[i], ax=axs[0, i])
                self.plotter.plot_episode(trials[len(trials) - i - 1], ax=axs[1, (4 - i)])
        if percentage_captured:
            self.plotter.plot_percentage_captured(trials)
        if animate[0]:
            self.plotter.animated_trial(trials[animate[1]])

    