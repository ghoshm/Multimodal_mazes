import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import patches
from matplotlib import colors
import matplotlib.animation as animation

class GridPlotter:
    def __init__(self, agent):
        self.agent = agent

    def plot_env(self, env, ax=None):
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
        for prey_location in prey_locations:
            prey_patch = patches.Circle(xy=(prey_location[1], prey_location[0]), radius=0.5, color='black', zorder=2)
            ax.add_patch(prey_patch)
            ax.imshow((colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:ultramarine'])(env[:, :, 0]) + colors.LinearSegmentedColormap.from_list('', ['white', 'xkcd:magenta'])(env[:, :, 1]))/2, interpolation='gaussian', zorder=0) 
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
            ax = self.plot_prey(env, prey_locations, ax)

        ax.set_title(f'{len(agent_path)} Steps')

        for i in range(len(agent_path)):
            trial_frac = i / self.agent.n_steps
            color=cm.get_cmap('viridis')(trial_frac)
            ax = self.plot_agent(ax, agent_location=agent_path[i], color = color)
        return ax

    def plot_training_progress(self, relative_trial_lengths):
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
        
    def reset(self):
        self.location = np.copy(self.reset_location)
        self.agent_direction = 0
        
    def sense_features(self, location):
        features = np.zeros((self.n_features))
        prey_location, scaled_distance, scaled_angle = self.closest_prey_features(location)
        features[0] = scaled_distance
        features[1] = scaled_angle
        features[2] = self.env[location[0] - 1, location[1], 0]   
        features[3] = self.env[location[0] + 1, location[1], 0]    
        features[4] = self.env[location[0], location[1] - 1, 0]   
        features[5] = self.env[location[0], location[1] + 1, 0]   
        features[6] = self.prey_pm
        # features[2] = self.env[location[0] - 1, location[1], 0]     # distance up
        # features[3] = self.env[location[0] + 1, location[1], 0]     # distance down 
        # features[4] = self.env[location[0], location[1] - 1, 0]     # distance left
        # features[5] = self.env[location[0], location[1] + 1, 0]     # distance right
        # features[6] = self.env[location[0] - 1, location[1] + 1, 0] # distance up-right
        # features[7] = self.env[location[0] - 1, location[1] - 1, 0] # distance up-left
        # features[8] = self.env[location[0] + 1, location[1] + 1, 0] # distance down-right
        # features[9] = self.env[location[0] + 1, location[1] - 1, 0] # distance down-left
        # features[10] = self.prey_pm
        return features
    
    def closest_prey_features(self, location):
        nearest_prey = min(self.prey_locations, key=lambda prey: np.linalg.norm(location - np.array(prey)))
        delta = location - nearest_prey[0]
        scaled_distance = np.linalg.norm(delta) / self.max_distance
        angle = np.arctan2(delta[1], delta[0])
        scaled_angle = angle / np.pi
        return nearest_prey, scaled_distance, scaled_angle
        
    def policy(self, action):
        next_location = self.location + self.actions[action]['delta']
        next_distance = self.closest_prey_features(next_location)[1]
        closest_distance_difference = next_distance - self.closest_prey_features(self.location)[1]
        
        if self.env[next_location[0], next_location[1], -1] == 1:
            if next_location in self.prey_locations:
                reward = 1000
            elif closest_distance_difference < 0:
                reward = self.cost_per_step + (self.max_distance * 10 / (next_distance * self.max_distance))
            else:
                reward = self.cost_per_step #- (self.max_distance * 10 / (next_distance * self.max_distance))

        else:
            next_location = np.copy(self.location)
            reward = self.cost_per_collision

        return next_location, reward
    
    def act(self, env, prey_locations, prey_pm):
        # self.env = np.array(env)
        # self.prey_locations = [np.array(prey) for prey in prey_locations]
        self.env = env
        self.prey_pm = prey_pm
        self.prey_locations = prey_locations
        
        action = self.epsilon_greedy_policy(self.epsilon, self.location)      
        next_location, _ = self.policy(action)
        self.location = next_location
        return next_location
        
    def training_act(self, env, prey_locations, prey_pm):
        # self.env = np.array(env)
        # self.prey_locations = [np.array(prey) for prey in prey_locations]
        self.env = env
        self.prey_pm = prey_pm
        self.prey_locations = prey_locations
        
        action = self.epsilon_greedy_policy(self.epsilon, self.location)        
        next_location, reward = self.policy(action)
        next_action = self.epsilon_greedy_policy(0, next_location)
        
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

    def epsilon_greedy_policy(self, epsilon, location):
        return np.random.randint(self.n_actions) if (np.random.rand() < epsilon) else np.argmax(self.Q_value(location))
    
    def update_parameter(self, parameter, decay_rate, parameter_min):
        return max(parameter * decay_rate, parameter_min)
    
    def minimum_trial_length(self, location, prey_locations):
        min_length = 0
        while len(prey_locations) > 0:
            nearest_prey = min(prey_locations, key=lambda reward: np.linalg.norm(location - np.array(reward)))
            delta = location - prey_locations[0]
            min_length += abs(delta[0]) + abs(delta[1])
            location = nearest_prey        
            prey_locations = np.delete(nearest_prey, np.argwhere(prey_locations == nearest_prey))
        return min_length
    
    def produce_plots(self, training_lengths, first_5_last_5, percentage_captured, animate, trials, trial_lengths):
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