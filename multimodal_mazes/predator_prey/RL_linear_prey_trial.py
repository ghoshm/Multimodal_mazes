import matplotlib.pyplot as plt
import numpy as np
import multimodal_mazes
from tqdm import tqdm
from scipy import signal

class RLPredatorTrial:
    """
    Handles the simulation of a single predator-prey trial. 

    Parameters:
    - width (int): Width of the environment grid.
    - height (int): Height of the environment grid.
    - agent (object): The agent object representing the predator.
    - sensor_noise_scale (float): Scale for sensor noise for the agent.
    - n_prey (int): Number of prey in the trial.
    - pk (int): Size of the kernel for sensory processing.
    - n_steps (int): Total number of steps in the trial.
    - scenario (str): Scenario type (e.g., 'Static', 'Dynamic').
    - case (str): Specific case to run within the scenario.
    - motion (str): Type of prey movement.
    - visible_steps (int): Number of steps for which prey is visible.
    - multisensory (str): Multisensory mode for the environment.
    - pc (float): Environmental persistence constant.
    - pm (float, optional): Prey movement probability.
    - pe (float, optional): Environmental cue probability.
    - log_env (bool): Flag to log environment data.

    Methods:
    - init_env(): Initializes the environment and agent.
    - init_preys(): Initializes prey based on the scenario and case.
    - run_training_trial(): Runs a training trial with RL for the agent.
    - run_trial(): Executes a test trial without RL training.
    - process_preys(): Updates prey locations and checks for prey capture.
    - emit_cues(): Emits sensory cues for the prey.
    - emit_noise(): Emits noise cues for prey in the environment.
    """

    def __init__(self, width, height, agent, sensor_noise_scale, n_prey, pk, n_steps, scenario, case, motion, visible_steps, multisensory, pc, pm=None, pe=None, log_env=True):
        self.width = width
        self.height = height
        self.agent = agent
        self.sensor_noise_scale = sensor_noise_scale
        self.n_prey = n_prey
        self.prey_counter = n_prey
        self.pk = pk
        self.pk_hw = pk // 2
        self.n_steps = n_steps
        self.scenario = scenario
        self.case = case
        self.motion = motion
        self.visible_steps = visible_steps
        self.multisensory = multisensory
        self.pc = pc
        self.pm = pm
        self.pe = pe
        self.log_env = log_env
        self.preys = []
        self.env_log = []
        self.env = None
        self.path = []
        self.init_env()

    def init_env(self):
        """
        Initializes the environment grid, including sensory channels and environment boundaries.
        """
        self.env = np.zeros((self.height, self.width, len(self.agent.channels) + 1))
        
        for r in range(self.height):
            for c in range(self.width):
                if r < 2:  
                    self.env[r, c, -1] = 1.0
                elif c >= (r - 1) and c < self.width - (r - 1):
                    self.env[r, c, -1] = 1.0

        self.env = np.pad(self.env, pad_width=((self.pk_hw, self.pk_hw), (self.pk_hw, self.pk_hw), (0, 0)))
        self.env_log.append(np.copy(self.env))
        self.agent.sensor_noise_scale = self.sensor_noise_scale
        self.agent.reset()
        self.init_preys()

    def init_preys(self):
        """
        Initializes prey objects based on the scenario, case, and motion parameters.
        Prey are positioned within the environment and their paths are initialized.
        """
        k1d = signal.windows.gaussian(self.pk, std=5)
        self.k2d = np.outer(k1d, k1d)
        k1d_noise = signal.windows.gaussian(self.pk//8, std=1)
        self.k2d_noise = np.outer(k1d_noise, k1d_noise)

        if self.scenario == 'Static':
            start_c = np.random.choice(range(self.width), size=self.n_prey, replace=False)
            direction = [0]
        else:
            possible_starts = [[self.width//2], [self.width-2, 1], [(self.width//4) - 2, ((3*self.width)//4) + 2], [self.width-5, 4]]
            choice = np.random.choice(range(2))
            directions = [-1, 1]
            if self.case == '4':
                direction = [directions[choice], directions[1 - choice]]
                start_c = [possible_starts[int(self.case) - 1][choice], possible_starts[int(self.case) - 1][1 - choice]]
            else:
                direction = [directions[choice]]
                start_c = [possible_starts[int(self.case) - 1][choice]] if len(possible_starts[int(self.case) - 1]) == 2 else [possible_starts[int(self.case) - 1][0]]

        for n in range(self.n_prey):
            prey = multimodal_mazes.PreyLinear(location=np.array([self.pk_hw, self.pk_hw + start_c[n]]), channels=[0, 0], scenario=self.scenario, pm=self.pm, motion=self.motion, direction=direction[n])
            prey.state = 1
            prey.path = [prey.location.copy()]

            if self.scenario != "Two Prey":
                prey.cues = (n % 2, ((n+1) % 2))
            else:
                if n == 0:
                    cue = np.random.choice(range(2))
                    prey.cues = (cue, 1 - cue)
                prey.cues = (1 - cue, cue)

            self.preys.append(prey)
    
    def run_training_trial(self):
        """
        Runs a training trial where the agent learns via reinforcement learning. 
        Returns data about the trial including agent path, rewards, and prey locations.
        """
        training_trial_data = {'path': [], 'rewards': [], 'prey_locations': [], 'env': []}

        for time in range(self.n_steps):
            self.env[:, :, :-1] *= self.pc
            prey_locations, prey_directions = self.process_preys(time)
            
            if self.prey_counter == 0:
                break
            if self.log_env:
                self.env_log.append(np.copy(self.env))

            visible = False if self.visible_steps < time else True
            location, reward = self.agent.training_act(self.env_log[-1], prey_locations, prey_directions, self.pm, visible) 
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
            prey_locations, prey_directions = self.process_preys(time)
            
            if self.prey_counter == 0:
                break
            if self.log_env:
                self.env_log.append(np.copy(self.env))

            location = self.agent.act(self.env_log[-1], prey_locations, prey_directions,  self.pm) 
            test_trial_data['path'].append(location)
            test_trial_data['prey_locations'].append(prey_locations)
        test_trial_data['prey'] = self.preys
        test_trial_data['env'] = self.env_log

        return test_trial_data

    def process_preys(self, time, move=True):
        """
        Processes prey movement, updates prey locations, and checks if prey has been captured by the agent.
        
        Parameters:
        - time (int): Current step in the simulation.
        
        Returns:
        - prey_locations (ndarray): Array of prey locations.
        - prey_directions (ndarray): Array of prey directions.
        """
        prey_locations = []
        prey_directions = []

        for prey in self.preys:
            if prey.state == 1:
                prey_locations.append(prey.location.copy())
                prey_directions.append(prey.direction)
                if np.array_equal(prey.location, self.agent.location):
                    prey.state = 0
                    self.prey_counter -= 1

                    if self.scenario == 'Two Prey':
                        self.prey_counter = 0
                else:
                    if self.scenario != "Static" and np.random.rand() < self.pm and move:
                        prey.move(self.env)
                    if prey.collision == 1:
                        self.prey_counter -= 1
                        continue
                    
                    prey.path.append(prey.location.copy())
                    self.emit_cues(prey, time)
                    # self.emit_noise(prey)

        return np.array(prey_locations), np.array(prey_directions)

    def emit_cues(self, prey, time):
        """
        Emits sensory cues from the prey based on the prey's position and scenario.

        Parameters:
        - prey (object): The prey object emitting cues.
        - time (int): The current time step in the simulation.
        """
        self.pk_hw = self.pk // 2
        r, c = prey.location
        cue_top = r - self.pk_hw
        cue_bottom = r + self.pk_hw
        cue_left = c - self.pk_hw
        cue_right = c + self.pk_hw

        if time <= self.visible_steps:
            if self.multisensory == "Balanced" and self.preys.index(prey) == 0:
                self.env[cue_top: cue_bottom, cue_left: cue_right, prey.cues[0]] += 0.5 * self.env[cue_top: cue_bottom, cue_left: cue_right, -1] * self.k2d[:cue_bottom - cue_top, :cue_right - cue_left]
                self.env[cue_top: cue_bottom, cue_left: cue_right, prey.cues[1]] += 0.5 * self.env[cue_top: cue_bottom, cue_left: cue_right, -1] * self.k2d[:cue_bottom - cue_top, :cue_right - cue_left]
            else:
                self.env[cue_top: cue_bottom, cue_left: cue_right, prey.cues[0]] += self.env[cue_top: cue_bottom, cue_left: cue_right, -1] * self.k2d[:cue_bottom - cue_top, :cue_right - cue_left]

    def emit_noise(self, prey):
        """
        Emits noise cues from the prey within the environment.
        
        Parameters:
        - prey (object): The prey object emitting noise.
        """
        self.pk_hw = self.pk // 2
        r, c = prey.location
        cue_top = r - self.pk_hw // 8
        cue_bottom = r + self.pk_hw // 8
        cue_left = c - self.pk_hw // 8
        cue_right = c + self.pk_hw // 8

        np.random.shuffle(self.k2d_noise.reshape(-1))
        self.env[cue_top: cue_bottom, cue_left: cue_right, prey.cues[1]] += self.k2d_noise[:cue_bottom - cue_top, :cue_right - cue_left]


class RLLinearPreyEvaluator:
    """
    Evaluates the performance of a predator agent across multiple trials.
    
    Parameters:
    - width (int): Width of the environment grid.
    - height (int): Height of the environment grid.
    - agent (object): The agent object representing the predator.
    - sensor_noise_scale (float): Scale for sensor noise for the agent.
    - n_prey (int): Number of prey in the trial.
    - pk (int): Size of the kernel for sensory processing.
    - n_steps (int): Total number of steps in the trial.
    - scenario (str): Scenario type (e.g., 'Static', 'Dynamic').
    - case (str): Specific case to run within the scenario.
    - motion (str): Type of prey movement.
    - visible_steps (int): Number of steps for which prey is visible.
    - multisensory (str): Multisensory mode for the environment.
    - pc (float): Environmental persistence constant.
    - pm (float, optional): Prey movement probability.
    - pe (float, optional): Environmental cue probability.
    
    Methods:
    - train_RL(): Trains the agent using reinforcement learning over multiple trials.
    - training_plots(): Generates training performance plots.
    - evaluate(): Evaluates agent performance on test trials.
    - calculate_success(): Calculates success metrics (prey captured and approached) in test trials.
    - calculate_optimal_length(): Computes the optimal length to capture prey based on case and movement probability.
    """

    def __init__(self, width, height, agent, sensor_noise_scale, n_prey, pk, n_steps, scenario, case, motion, visible_steps, multisensory, pc, pm=None, pe=None):
        self.width = width
        self.height = height
        self.agent = agent
        self.sensor_noise_scale = sensor_noise_scale
        self.n_prey = n_prey
        self.pk = pk
        self.n_steps = n_steps
        self.scenario = scenario
        self.case = case
        self.motion = motion
        self.visible_steps = visible_steps
        self.multisensory = multisensory
        self.pc = pc
        self.pm = pm
        self.pe = pe

    def train_RL(self, training_trials, curriculum=False):
        """
        Trains the predator agent using reinforcement learning over multiple trials.
        
        Parameters:
        - training_trials (int): Number of training trials to execute.
        - curriculum (bool): Whether to use curriculum learning.
        """
        self.training_trials = {}
        self.trial_lengths = []
        pm_range = (0, 1) if not curriculum else (0, 0.1)
        visible_steps_range = (1, 30) if not curriculum else (25, 30)
        total_length = 0
        optimal_length = 0
        disappearing = self.visible_steps is None

        for trial in tqdm(range(training_trials)):
            # disappearing = False
            if self.scenario == 'Constant':
                self.case = str(np.random.randint(1, 4))
                # self.pm = np.random.rand()
                self.pm = np.random.uniform(pm_range[0], pm_range[1])
            # if self.visible_steps is None and trial == 0:
                # disappearing = True

            self.visible_steps = np.random.randint(visible_steps_range[0], visible_steps_range[1]) if disappearing else self.visible_steps
                
            self.training_trial = RLPredatorTrial(width=self.width, height=self.height, agent=self.agent, sensor_noise_scale=self.sensor_noise_scale, n_prey=self.n_prey, pk=self.pk, n_steps=self.n_steps, scenario=self.scenario, case=self.case, motion=self.motion, visible_steps=self.visible_steps, multisensory=self.multisensory, pc=self.pc, pm=self.pm, pe=self.pe)
            training_trial_data = self.training_trial.run_training_trial()
            self.training_trials[trial] = training_trial_data
            self.training_trials[trial]['prey_states'] = [prey.state for prey in self.training_trial.preys]
            self.trial_lengths.append(len(training_trial_data['path']))
            self.agent.cost_per_step = self.agent.update_parameter(self.agent.cost_per_step, 1.1, -50)
            
            # if trial % 5 == 0:
            #     self.agent.update_target_network()

            if curriculum:
                optimal_length += self.calculate_optimal_length(self.case, self.pm, training_trial_data['prey_locations'], training_trial_data['path'][0])
                total_length += len(training_trial_data['path'])

                if trial % 50 == 0 and trial != 0:
                    if (optimal_length * 0.85 <= total_length <= optimal_length * 1.15) and pm_range[1] == 1:
                        print(f'End training, Trial number: {trial}')
                        break
                    elif (optimal_length * 0.85 <= total_length <= optimal_length * 1.15) and pm_range[1] != 1:
                        print(f'End training stage with pm range: {pm_range}, Trial number: {trial}')
                        pm_range = (pm_range[0], min(1, pm_range[1] + 0.3))
                        visible_steps_range = (max(1, visible_steps_range[0] - 6), visible_steps_range[1]) if disappearing else (25, 30)
                        
                    total_length = 0
                    optimal_length = 0
     
    def training_plots(self, training_lengths=False, first_5_last_5=False, percentage_captured=False, animate=[False, None]):
        """
        Generates plots visualizing training performance. 
        Can display trial lengths, capture percentages, and animations.
        
        Parameters:
        - training_lengths (bool): Plot the lengths of training trials.
        - first_5_last_5 (bool): Compare performance between the first and last 5 trials.
        - percentage_captured (bool): Plot percentage of prey captured.
        - animate (list): Animation options for trial visualization.
        """
        self.agent.produce_plots(training_lengths=training_lengths, first_5_last_5=first_5_last_5, percentage_captured=percentage_captured, animate=animate, trials=self.training_trials, trial_lengths=self.trial_lengths)

    def evaluate(self, n_trials, case, pm, visible_steps=100):
        """
        Runs evaluation trials and returns the success rate of prey capture.
        
        Parameters:
        - n_trials (int): Number of evaluation trials to run.
        - case (str): Specific case to test.
        - pm (float): Prey movement probability.
        - visible_steps (int): Number of steps for which prey is visible.
        
        Returns:
        - test_trials (dict): Data of the test trials.
        - captured (float): Percentage of prey captured.
        - approached (float): Percentage of prey approached.
        """
        self.test_trials = {}
        self.visible_steps = visible_steps
        
        for trial in range(n_trials):
            test_trial = RLPredatorTrial(width=self.width, height=self.height, agent=self.agent, sensor_noise_scale=self.sensor_noise_scale, n_prey=self.n_prey, pk=self.pk, n_steps=self.n_steps, scenario=self.scenario, case=case, motion=self.motion, visible_steps=self.visible_steps, multisensory=self.multisensory, pc=self.pc, pm=pm, pe=self.pe)
            test_trial_data = test_trial.run_trial()
            self.test_trials[trial] = test_trial_data
            
        captured, approached = self.calculate_success(self.test_trials, n_trials)
        return self.test_trials, captured, approached

    def calculate_success(self, trials, n_trials):
        """
        Calculates the percentage of prey captured and approached across trials.
        
        Parameters:
        - trials (dict): Dictionary containing trial data.
        - n_trials (int): Number of trials to evaluate.

        Returns:
        - captured (float): Percentage of prey captured.
        - approached (float): Percentage of prey approached.
        """
        total = 0
        captured = 0
        approached = 0
        
        for trial in range(n_trials):
            prey_captured = [prey.location.copy() for prey in trials[trial]['prey'] if prey.state == 0]
            prey_not_captured = [prey.location.copy() for prey in trials[trial]['prey'] if prey.state == 1]
            last_agent_location = trials[trial]['path'][-1]
            
            for prey_location in prey_not_captured:
                if (last_agent_location == prey_location + np.array([1, 0])).all() or (last_agent_location == prey_location + np.array([0, 1])).all() or (last_agent_location == prey_location + np.array([0, -1])).all(): # or (last_agent_location == prey_location + np.array([1, 1])).all() or (last_agent_location == prey_location + np.array([1, -1])).all():
                    approached += 1

            total += len(prey_captured) + len(prey_not_captured)
            captured += len(prey_captured)
            approached += len(prey_captured)
        captured = 100 * captured / total
        approached = 100 * approached / total
        return captured, approached
    
    def calculate_optimal_length(self, case, pm, prey_locations, start_position):
        """
        Calculates the optimal path length to capture prey based on case and prey movement.
        
        Parameters:
        - case (str): Specific case to evaluate.
        - pm (float): Prey movement probability.
        - prey_locations (ndarray): Initial locations of the prey.
        - start_position (ndarray): Starting position of the agent.
        
        Returns:
        - optimal_length (int): Computed optimal length for prey capture.
        """
        optimal_length = 0
    
        for prey in prey_locations[0]:
            distance =  abs(start_position[0] - prey[0]) + abs(start_position[0] - prey[0])
            optimal_length += (distance + int(distance * pm)) if case != "2" else (distance - int(distance * pm))

        return optimal_length
    