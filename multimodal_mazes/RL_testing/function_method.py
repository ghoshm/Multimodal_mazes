import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import patches
from matplotlib import colors

class GridPlotter:
    def __init__(self, agent):
        self.agent = agent

    def plot_env(self, env, ax=None):
        height, width = env[:, :, -1].shape
        if ax is None:
            fig, ax = plt.subplots(figsize=(0.1 * width, 0.1 * height))
        ax.imshow((colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:ultramarine'])(env[:, :, 0]) + colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:magenta'])(self.agent.env[:, :, 1]))/2, interpolation='gaussian', zorder=0) 
        ax.imshow(1 - env[:, :, -1], cmap=cm.binary, alpha=0.25, zorder=1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim([self.agent.pk_hw - 1, width - self.agent.pk_hw])
        ax.set_ylim([height - self.agent.pk_hw, self.agent.pk_hw - 1])
        return ax 

    def plot_prey(self, prey_locations, ax):
        for prey_location in prey_locations:
            prey_patch = patches.Circle(xy=(prey_location[1], prey_location[0]), radius=0.5, color='black', zorder=2)
            ax.add_patch(prey_patch)
        return ax

    def plot_agent(self, ax, agent_location, color):
        agent_patch = patches.Rectangle((agent_location[1] - 0.5, agent_location[0] - 0.5), 0.8, 0.8, color=color, zorder=3)
        ax.add_patch(agent_patch)
        return ax

    def plot_episode(self, trial_data, ax=None):
        agent_path = trial_data['path']
        prey_locations = trial_data['prey_locations'][0]
        env = trial_data['env'][0]

        if ax is None:
            ax = self.plot_env(env)
            ax = self.plot_prey(prey_locations, ax)

        ax.set_title(f'{len(agent_path)} Steps')

        for i in range(len(agent_path)):
            trial_frac = i / self.agent.n_steps
            color=cm.get_cmap('viridis')(trial_frac)
            ax = self.plot_agent(ax, agent_location=agent_path[i], color = color)
        return ax

    def plot_training_progress(self, relative_trial_lengths, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        n_trials = len(relative_trial_lengths)
        smoothed_episode_lengths = [np.mean(relative_trial_lengths[max(0, i - 100) : i + 1]) for i in range(n_trials)]
        ax.scatter(np.arange(n_trials), relative_trial_lengths, linewidth=0, alpha=0.5, c='C0', label='Episode length')
        ax.plot(np.arange(len(smoothed_episode_lengths)), smoothed_episode_lengths, color='k', linestyle='--', linewidth=0.5, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Relative Length')
        ax.legend()
        ax.set_title('Training Progress')
        return ax


class QLearnerAgent:
    def __init__(self, pk_hw, channels, actions, location, sensor_noise_scale, n_steps, n_features, cost_per_step, cost_per_collision, alpha, epsilon, gamma):
        self.pk_hw = pk_hw
        self.channels = np.array(channels)
        self.actions = actions
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
        
    def reset(self):
        self.agent_direction = int(0)
        
    def sense_features(self, location):
        features = np.zeros((self.n_features))
        scaled_distance, scaled_angle = self.closest_prey_features(location)
        features[0] = self.env[location[0] - 1, location[1], 0]
        features[1] = self.env[location[0] + 1, location[1], 0]
        features[2] = self.env[location[0], location[1] - 1, 0]
        features[3] = self.env[location[0], location[1] + 1, 0]
        features[4] = scaled_distance
        features[5] = scaled_angle
        # print(features)
        return features
    
    def closest_prey_features(self, location):
        nearest_prey = min(self.prey_locations, key=lambda reward: np.linalg.norm(location - np.array(reward)))
        delta = tuple(np.array(location) - np.array(nearest_prey[0]))
        scaled_distance = float((abs(delta[0]) + abs(delta[1])) / self.max_distance)
        angle = float(np.arctan2(delta[1], delta[0]))
        scaled_angle = angle / (2 * np.pi) if angle >= 0 else (angle + 2 * np.pi) / (2 * np.pi)
        return scaled_distance, scaled_angle
        
    def policy(self, action):
        next_location = tuple(np.array(self.location) + np.array(self.actions[action]['delta']))
        next_distance = self.closest_prey_features(next_location)[0]
        closest_distance_difference = next_distance - self.closest_prey_features(self.location)[0]
        
        if self.env[next_location[0], next_location[1], -1] == 1:
            if next_location in self.prey_locations:
                reward = 1000
            elif closest_distance_difference < 0:
                reward = self.max_distance * 10 / (next_distance * self.max_distance) # + self.cost_per_step
            else:
                # reward = self.cost_per_step
                reward = - (self.max_distance * 10 / (next_distance * self.max_distance))

        else:
            next_location = self.location
            reward = float(self.cost_per_collision)

        return next_location, reward
    
    def act(self, env, prey_locations):
        self.env = env
        self.prey_locations = prey_locations
        
        action = int(self.epsilon_greedy_policy(self.location))        
        next_location, _ = self.policy(action)
        self.location = next_location
        # self.learn(self.location, next_location, action, next_action, reward, decaying_alpha)
        
    def training_act(self, env, prey_locations):
        self.env = env
        self.prey_locations = prey_locations
        
        action = int(self.epsilon_greedy_policy(self.location))        
        next_location, reward = self.policy(action)
        next_action = int(self.epsilon_greedy_policy(next_location))
        
        self.learn(self.location, next_location, action, next_action, reward, self.alpha)
        self.location = next_location

        self.alpha = self.update_parameter(self.alpha, 0.99, 0.05)
        self.epsilon = self.update_parameter(self.epsilon, 0.99, 0.01)

        return next_location, reward
        
    def learn(self, location, next_location, action, next_action, reward, alpha):
        Q = float(self.Q_value(location)[action])
        Q_next =  float(self.Q_value(next_location)[next_action])
        TD_error = float(reward) + (float(self.gamma) * float(Q_next)) - float(Q)
        self.theta[:, action] += float(alpha) * float(TD_error) * self.sense_features(location)            
        return TD_error

    def Q_value(self, location):
        return np.dot(self.sense_features(location), self.theta)

    def epsilon_greedy_policy(self, location):
        return int(np.random.randint(self.n_actions)) if (np.random.rand() < self.epsilon) else int(np.argmax(self.Q_value(location)))
    
    def update_parameter(self, parameter, decay_rate, parameter_min):
        return max(parameter * decay_rate, parameter_min)
    
    def minimum_distance(self, agent_start_location, prey_locations):
        delta = tuple(np.array(agent_start_location) - np.array(prey_locations[0]))
        min_distance = abs(delta[0]) + abs(delta[1])
        for i in range(len(prey_locations[:-1])):
            delta = tuple(np.array(agent_start_location) - np.array(prey_locations[0]))
            min_distance += abs(delta[0]) + abs(delta[1])
        return min_distance
    
    def produce_training_plots(self, training, first_5_last_5, training_trials, trial_lengths):
        self.plotter = GridPlotter(self)
        if training:
            min_distances = [self.minimum_distance(training_trials[i]['path'][0], training_trials[i]['prey_locations'][0]) for i in range(len(trial_lengths))]
            relative_trial_lengths = [trial_lengths[i] / min_distances[i] for i in range(len(trial_lengths))]
            self.plotter.plot_training_progress(relative_trial_lengths=relative_trial_lengths)
        if first_5_last_5:
            fig, axs = plt.subplots(2, 5, figsize=(10, 4))
            for i in range(5):
                axs[0, i] = self.plotter.plot_env(training_trials[i]['env'][1], ax=axs[0, i])
                axs[1, (4 - i)] = self.plotter.plot_env(training_trials[len(training_trials) - i - 1]['env'][1], ax=axs[1, (4 - i)])
                axs[0, i] = self.plotter.plot_prey(training_trials[i]['prey_locations'][0], ax=axs[0, i])
                axs[1, (4 - i)] = self.plotter.plot_prey(training_trials[len(training_trials) - i - 1]['prey_locations'][0], ax=axs[1, (4 - i)])
                
                self.plotter.plot_episode(training_trials[i], ax=axs[0, i])
                self.plotter.plot_episode(training_trials[len(training_trials) - i - 1], ax=axs[1, (4 - i)])