import matplotlib.pyplot as plt
import numpy as np
import multimodal_mazes
from tqdm import tqdm
from scipy import signal

class PredatorTrialContinuous:
    """
    Simulates a single predator-prey trial in a continuous environment.

    Parameters:
        width (int): Width of the environment grid.
        height (int): Height of the environment grid.
        pk_hw (int): Half the size of the kernel for sensory processing.
        agent (object): The agent object representing the predator.
        sensor_noise (float): Scale for sensor noise for the agent.
        n_prey (int): Number of prey in the trial.
        capture_radius (float): Radius within which prey is considered captured.
        n_steps (int): Total number of steps in the trial.
        visible_steps (int): Number of steps for which prey is visible.
        scenario (str): Scenario type (e.g., 'Static', 'Dynamic').
        motion (str): Type of prey movement.
        case (str): Specific case to run within the scenario.
        multisensory (str): Multisensory mode for the environment.
        speed (float): Prey speed.
        pe (float): Environmental cue probability.
        pc (float): Environmental persistence constant.
        log_env (bool): Flag to log environment data (default False).

    Attributes:
        env (np.ndarray): The environment grid.
        env_log (list): List to log environment data over time.
        preys (list): List of prey objects.
        path (list): List to record agent's path.
    """

    def __init__(self, width, height, pk_hw, agent, sensor_noise, n_prey, 
                 capture_radius, n_steps, visible_steps, scenario, motion, 
                 case, multisensory, speed, pe, pc):
        self.width = width
        self.height = height
        self.pk_hw = pk_hw
        
        self.agent = agent
        self.sensor_noise = sensor_noise
    
        self.n_prey = n_prey
        self.prey_counter = n_prey
        self.capture_radius = capture_radius
        
        self.n_steps = n_steps
        self.visible_steps = visible_steps
        self.scenario = scenario
        self.motion = motion
        self.case = case
        self.multisensory = multisensory
        
        self.speed = speed
        self.pe = pe
        self.pc = pc
        
        self.env = None
        self.env_log = []
        self.path = []
        self.preys = []
        self.init_env()

    def init_env(self):
        """
        Initializes the environment grid, including sensory channels and environment boundaries.
        """
        self.env = np.zeros((self.height, self.width, len(self.agent.channels) + 1))
        
        r = np.arange(self.height)
        c = np.arange(self.width)
        R, C = np.meshgrid(r, c, indexing='ij')

        mask = (R < 2) | ((R >= 2) & (C >= (R - 1)) & (C < (self.width - (R - 1))))
        self.env[mask, -1] = 1.0

        pad_width = ((self.pk_hw, self.pk_hw), (self.pk_hw, self.pk_hw), (0, 0))
        self.env = np.pad(self.env, pad_width=pad_width)

        self.env_log.append(np.copy(self.env))
        
        self.agent.sensor_noise = self.sensor_noise
        self.agent.reset()

        self.init_preys()

    def init_preys(self):
        """
        Initializes prey objects based on the scenario, case, and motion parameters.
        Prey are positioned within the environment and their paths are initialized.
        """
        pk = self.pk_hw * 2
        k1d = signal.windows.gaussian(pk, std=5)
        self.k2d = np.outer(k1d, k1d)

        k1d_noise = signal.windows.gaussian(pk//8, std=1)
        self.k2d_noise = np.outer(k1d_noise, k1d_noise)

        case_positions = {
            '1': [self.width / 2],
            '2': [self.width - 2, 1],
            '3': [(self.width / 4) - 2, ((3 * self.width) / 4) + 2],
            '4': [self.width - 5, 4],
        }

        if self.scenario == 'Static':
            x0 = [np.random.uniform(0.5, self.width - 0.5)]
            speed = [0]
        else:
            case_x0s = case_positions.get(self.case)
            choice = np.random.choice([0, 1]) if len(case_x0s) > 1 else 0
            speeds = [-self.speed, self.speed]
            
            if self.case == '4':
                speed = [speeds[choice], speeds[1-choice]]
                x0 = [case_x0s[choice], case_x0s[1 - choice]]
            else:
                speed = [speeds[choice]]
                x0 = [case_x0s[choice]]

        for n in range(self.n_prey):
            prey = multimodal_mazes.PreyContinuous(location=np.array([self.pk_hw, self.pk_hw + x0[n]]), scenario=self.scenario, motion=self.motion, channels=[0, 0], speed=speed[n])
            prey.state = 1
            prey.path = [prey.location.copy()]

            if self.scenario != "Two Prey":
                prey.cues = (n % 2, ((n + 1) % 2))
            else:
                if n == 0:
                    cue = np.random.choice([0, 1])
                    prey.cues = (cue, 1 - cue)
                prey.cues = (1 - cue, cue)

            self.preys.append(prey)
    
    def run_training_trial(self):
        """
        Runs a training trial where the agent learns via reinforcement learning. 
        Returns data about the trial including agent path, rewards, and prey locations.
        """
        training_trial_data = {'path': [self.agent.location], 'rewards': [], 'prey_locations': [], 'env': []}

        for time in range(self.n_steps):
            self.env[:, :, :-1] *= self.pc
            prey_locations, prey_speeds = self.process_preys(time)
            self.env_log.append(np.copy(self.env))
            
            if self.prey_counter == 0:
                break
                
            visible = time <= self.visible_steps
            location, reward = self.agent.act(self.env, prey_locations, prey_speeds, visible, training=True) 
            training_trial_data['path'].append(location)
            training_trial_data['rewards'].append(reward)
            training_trial_data['prey_locations'].append(prey_locations)
        
        training_trial_data['env'] = self.env_log

        return training_trial_data
    
    def run_trial(self):
        """
        Runs a test trial without learning. 
        Returns data about the trial including agent path and prey locations.
        """
        test_trial_data = {'path': [], 'preys': [], 'prey_locations': [], 'env': []}

        for time in range(self.n_steps):
            self.env[:, :, :-1] *= self.pc
            prey_locations, prey_speeds = self.process_preys(time)
            self.env_log.append(np.copy(self.env))
            
            if self.prey_counter == 0:
                break
                
            location, _ = self.agent.act(self.env, prey_locations, prey_speeds) 
            test_trial_data['path'].append(location)
            test_trial_data['prey_locations'].append(prey_locations)
        
        test_trial_data['prey'] = self.preys
        test_trial_data['env'] = self.env_log

        return test_trial_data

    def process_preys(self, time):
        """
        Processes prey movement, updates prey locations, and checks if prey has been captured by the agent.
        
        Parameters:
        - time (int): Current step in the simulation.
        
        Returns:
        - prey_locations (ndarray): Array of prey locations.
        - prey_directions (ndarray): Array of prey directions.
        """
        prey_locations = []
        prey_speeds = []

        for prey in self.preys:
            if prey.state == 1:
                prey_locations.append(prey.location.copy())
                prey_speeds.append(prey.speed)
                
                if np.linalg.norm(prey.location - self.agent.location) < self.capture_radius:
                    prey.state = 0
                    self.prey_counter -= 1

                    if self.scenario == 'Two Prey':
                        self.prey_counter = 0
                else:
                    if self.scenario != "Static":
                        prey.move(self.env)
                    
                    if prey.collision == 1:
                        self.prey_counter -= 1
                        continue
                    
                    prey.path.append(prey.location.copy())
                    
                    if time < self.visible_steps:
                        self.emit_cues(prey)
                        
                    # self.emit_noise(prey)

        return np.array(prey_locations), np.array(prey_speeds)

    def emit_cues(self, prey):
        """
        Emits sensory cues from the prey based on the prey's position and scenario.

        Parameters:
        - prey (object): The prey object emitting cues.
        """
        r, c = prey.location

        env_height, env_width = self.env.shape[:2]
        
        cue_top = int(r - self.pk_hw)
        cue_bottom = int(r + self.pk_hw)
        cue_left = int(c - self.pk_hw)
        cue_right = int(c + self.pk_hw)

        env_top = max(0, cue_top)
        env_bottom = min(env_height, cue_bottom)
        env_left = max(0, cue_left)
        env_right = min(env_width, cue_right)

        k_top = env_top - cue_top
        k_bottom = k_top + (env_bottom - env_top)
        k_left = env_left - cue_left
        k_right = k_left + (env_right - env_left)

        env_slice = self.env[env_top:env_bottom, env_left:env_right]
        kernel_slice = self.k2d[k_top:k_bottom, k_left:k_right]

        if self.multisensory == "Balanced" and self.preys.index(prey) == 0:
            env_slice[..., prey.cues[0]] += 0.5 * env_slice[..., -1] * kernel_slice
            env_slice[..., prey.cues[1]] += 0.5 * env_slice[..., -1] * kernel_slice
        else:
            env_slice[..., prey.cues[0]] += env_slice[..., -1] * kernel_slice

    def emit_noise(self, prey):
        """
        Emits sensory cues from the prey based on the prey's position and scenario.

        Parameters:
        - prey (object): The prey object emitting cues.
        """
        num_noise_points = np.random.randint(2, 10)
        
        for _ in range(num_noise_points):
            nx = int(self.pk_hw + np.random.uniform(0, self.width))
            ny = int(self.pk_hw + np.random.uniform(0, self.height))

            cue_top = int(ny - self.pk_hw // 8)
            cue_bottom = int(ny + self.pk_hw // 8)
            cue_left = int(nx - self.pk_hw // 8)
            cue_right = int(nx + self.pk_hw // 8)

            env_height, env_width = self.env.shape[:2]
            env_top = max(0, cue_top)
            env_bottom = min(env_height, cue_bottom)
            env_left = max(0, cue_left)
            env_right = min(env_width, cue_right)

            k_top = env_top - cue_top
            k_bottom = k_top + (env_bottom - env_top)
            k_left = env_left - cue_left
            k_right = k_left + (env_right - env_left)

            if env_bottom <= env_top or env_right <= env_left:
                continue

            np.random.shuffle(self.k2d_noise.reshape(-1))

            env_slice = self.env[env_top:env_bottom, env_left:env_right, prey.cues[1]]
            noise_slice = self.k2d_noise[k_top:k_bottom, k_left:k_right]

            if env_slice.shape != noise_slice.shape:
                noise_slice = noise_slice[:env_slice.shape[0], :env_slice.shape[1]]

            env_slice += noise_slice


class LinearPreyEvaluatorContinuous:
    """
    Evaluates the performance of a predator agent across multiple trials.
    
    arameters:
        width (int): Width of the environment grid.
        height (int): Height of the environment grid.
        pk_hw (int): Half the size of the kernel for sensory processing.
        agent (object): The agent representing the predator.
        sensor_noise (float): Scale for sensor noise for the agent.
        n_prey (int): Number of prey in each trial.
        capture_radius (float): Radius within which prey is considered captured.
        n_steps (int): Total number of steps in each trial.
        visible_steps (int): Number of steps for which prey is visible.
        scenario (str): Scenario type (e.g., 'Static', 'Dynamic').
        motion (str): Type of prey movement.
        case (str): Specific case to run within the scenario.
        multisensory (str): Multisensory mode for the environment.
        speed (float): Prey speed.
        pe (float): Environmental cue probability.
        pc (float): Environmental persistence constant.
    
    Methods:
    - train_RL(): Trains the agent using reinforcement learning over multiple trials.
    - training_plots(): Generates training performance plots.
    - evaluate(): Evaluates agent performance on test trials.
    - calculate_success(): Calculates success metrics (prey captured and approached) in test trials.
    - optimal_trial_length(): Computes the optimal length to capture prey based on case and movement probability.
    """

    def __init__(self, width, height, pk_hw, agent, sensor_noise, n_prey, 
                 capture_radius, n_steps, visible_steps, scenario, motion, 
                 case, multisensory, speed, pe, pc):
        self.width = width
        self.height = height
        self.pk_hw = pk_hw
        
        self.agent = agent
        self.sensor_noise = sensor_noise
    
        self.n_prey = n_prey
        self.capture_radius = capture_radius
        
        self.n_steps = n_steps
        self.visible_steps = visible_steps
        self.scenario = scenario
        self.motion = motion
        self.case = case
        self.multisensory = multisensory
        
        self.speed = speed
        self.pe = pe
        self.pc = pc

        self.training_trials = {}
        self.test_trials = {}
        self.trial_lengths = []

    def train_RL(self, training_trials, curriculum=False):
        """
        Trains the predator agent using reinforcement learning over multiple trials.
        
        Parameters:
        - training_trials (int): Number of training trials to execute.
        - curriculum (bool): Whether to use curriculum learning.
        """
        speed_range = (0, 1) if not curriculum else (0, 0.1)
        visible_steps_range = (1, 30) if not curriculum else (25, 30)
        total_length = 0
        optimal_length = 0
        disappearing = self.visible_steps is None

        for trial in tqdm(range(training_trials)):
            if self.scenario == 'Constant':
                self.case = str(np.random.randint(1, 4))
                self.speed = np.random.uniform(speed_range[0], speed_range[1])
            self.visible_steps = np.random.randint(*visible_steps_range) if disappearing else self.visible_steps
            
            training_trial_instance = PredatorTrialContinuous(self.width, self.height, self.pk_hw, self.agent, self.sensor_noise, self.n_prey, self.capture_radius, self.n_steps, self.visible_steps, self.scenario, self.motion, self.case, self.multisensory, self.speed, self.pe, self.pc)
            training_trial_data = training_trial_instance.run_training_trial()
            
            self.training_trials[trial] = training_trial_data
            self.training_trials[trial]['prey_states'] = [prey.state for prey in training_trial_instance.preys]
            self.trial_lengths.append(len(training_trial_data['path']))
            
            if curriculum:
                optimal_length += self.optimal_trial_length(self.case, self.speed, training_trial_data['prey_locations'], training_trial_data['path'][0])
                total_length += len(training_trial_data['path'])

                if trial % 50 == 0 and trial != 0:
                    optimal_range = optimal_length * 0.85 <= total_length <= optimal_length * 1.15
                    if optimal_range and speed_range[1] == 1:
                        print(f'End training, Trial number: {trial}')
                        break
                    elif optimal_range and speed_range[1] < 1:
                        print(f'Advancing curriculum at trial {trial}')
                        speed_range = (speed_range[0], min(1, speed_range[1] + 0.3))
                        visible_steps_range = (max(1, visible_steps_range[0] - 6), visible_steps_range[1]) if disappearing else (25, 30)
                        
                    total_length = 0
                    optimal_length = 0
     
    def training_plots(self, plot_training_lengths=False, plot_first_5_last_5=False, plot_percentage_captured=False, animate=[False, None]):
        """
        Generates plots visualizing training performance. 
        Can display trial lengths, capture percentages, and animations.
        
        Parameters:
        - training_lengths (bool): Plot the lengths of training trials.
        - first_5_last_5 (bool): Compare performance between the first and last 5 trials.
        - percentage_captured (bool): Plot percentage of prey captured.
        - animate (list): Animation options for trial visualization.
        """
        self.agent.produce_plots(
            plot_training_lengths=plot_training_lengths, 
            plot_first_5_last_5=plot_first_5_last_5, 
            plot_percentage_captured=plot_percentage_captured, 
            animate=animate, 
            trials=self.training_trials, 
            trial_lengths=self.trial_lengths, 
            pk_hw=self.pk_hw, 
            n_steps=self.n_steps
        )

    def evaluate(self, n_trials, case, speed, visible_steps=100):
        """
        Runs evaluation trials and returns the success rate of prey capture.
        
        Parameters:
        - n_trials (int): Number of evaluation trials to run.
        - case (str): Specific case to test.
        - speed (float): Prey speed.
        - visible_steps (int): Number of steps for which prey is visible.
        
        Returns:
        - test_trials (dict): Data of the test trials.
        - captured (float): Percentage of prey captured.
        - approached (float): Percentage of prey approached.
        """
        self.case = case
        self.speed = speed
        self.visible_steps = visible_steps
        
        for trial in range(n_trials):
            test_trial_instance = PredatorTrialContinuous(self.width, self.height, self.pk_hw, self.agent, self.sensor_noise, self.n_prey, self.capture_radius, self.n_steps, self.visible_steps, self.scenario, self.motion, self.case, self.multisensory, self.speed, self.pe, self.pc)
            self.test_trials[trial] = test_trial_instance.run_trial()
            
        captured, approached = self.calculate_success(n_trials)
        return self.test_trials, captured, approached

    def calculate_success(self):
        """
        Calculates the percentage of prey captured and approached across trials.
        
        Returns:
        - captured (float): Percentage of prey captured.
        - approached (float): Percentage of prey approached.
        """
        total = 0
        captured = 0
        approached = 0
        
        for trial_data in self.test_trials.values():
            preys = trial_data['prey']
            agent_last_location = trial_data['path'][-1]

            for prey in preys:
                total += 1
                prey_location = prey.location.copy()
                distance = np.linalg.norm(agent_last_location - prey_location)

                if prey.state == 0:
                    captured += 1
                elif distance < self.capture_radius + 0.5:
                    approached += 1
            # prey_captured = [prey.location.copy() for prey in trials[trial]['prey'] if prey.state == 0]
            # prey_not_captured = [prey.location.copy() for prey in trials[trial]['prey'] if prey.state == 1]
            # last_agent_location = trials[trial]['path'][-1]
            
            # for prey_location in prey_not_captured:
            #     if np.linalg.norm(last_agent_location - prey_location) < self.capture_radius + 0.5:
            #         approached += 1

            # total += len(prey_captured) + len(prey_not_captured)
            # captured += len(prey_captured)
            # approached += len(prey_captured)

        captured = 100 * captured / total
        approached = 100 * approached / total
        return captured, approached
    
    def optimal_trial_length(self, case, speed, prey_locations, start_position):
        """
        Calculates the optimal path length to capture prey based on case and prey movement.
        
        Parameters:
        - case (str): Specific case to evaluate.
        - speed (float): Prey speed.
        - prey_locations (ndarray): Initial locations of the prey.
        - start_position (ndarray): Starting position of the agent.
        
        Returns:
        - optimal_length (int): Computed optimal length for prey capture.
        """
        optimal_length = 0
    
        for prey in prey_locations[0]:
            distance = np.linalg.norm(start_position - prey)
            optimal_length += int(np.ciel((distance + (distance * speed))/2)) if case != "2" else int(np.ceil((distance - (distance * speed))/2))

        return optimal_length