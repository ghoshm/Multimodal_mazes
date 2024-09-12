import matplotlib.pyplot as plt
import numpy as np
import multimodal_mazes
from tqdm import tqdm
from scipy import signal

class PredatorTrial:
    def __init__(self, width, height, agnt, sensor_noise_scale, n_prey, pk, n_steps, scenario, case, motion, visible_steps, multisensory, pc, pm=None, pe=None, log_env=True):
        self.width = width
        self.height = height
        self.agnt = agnt
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
        self.env = np.zeros((self.height, self.width, len(self.agnt.channels) + 1))
        
        for r in range(self.height):
            for c in range(self.width):
                if r < 2:  
                    self.env[r, c, -1] = 1.0
                elif c >= (r - 1) and c < self.width - (r - 1):
                    self.env[r, c, -1] = 1.0

        self.env = np.pad(self.env, pad_width=((self.pk_hw, self.pk_hw), (self.pk_hw, self.pk_hw), (0, 0)))
        self.env_log.append(np.copy(self.env))
        self.agnt.sensor_noise_scale = self.sensor_noise_scale
        self.agnt.reset()
        self.init_preys()

    def init_preys(self):
        k1d = signal.windows.gaussian(self.pk, std=6)
        self.k2d = np.outer(k1d, k1d)
        k1d_noise = signal.windows.gaussian(self.pk//8, std=1)
        self.k2d_noise = np.outer(k1d_noise, k1d_noise)

        if self.scenario == 'Static':
            start_c = np.random.choice(range(self.width), size=self.n_prey, replace=False)
            direction = [0]
        else:
            possible_starts = [[self.width//2], [self.width-1, 0], [(self.width//4), ((3*self.width)//4)], [self.width-5, 4]]
            choice = np.random.choice(range(2))
            directions = [-1, 1]
            if self.case == '4':
                direction = [directions[choice], directions[1 - choice]]
                start_c = [possible_starts[int(self.case) - 1][choice], possible_starts[int(self.case) - 1][1 - choice]]
            else:
                direction = [directions[choice]]
                start_c = [possible_starts[int(self.case) - 1][choice]] if len(possible_starts[int(self.case) - 1]) == 2 else [possible_starts[int(self.case) - 1][0]]

        for n in range(self.n_prey):
            prey = multimodal_mazes.PreyLinear(location=np.array([self.pk_hw, self.pk_hw + start_c[n]]), channels=[0, 0], scenario=self.scenario, motion=self.motion, direction=direction[n])
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
        training_trial_data = {'path': [], 'rewards': [], 'prey_locations': [], 'env': []}
        
        for time in range(self.n_steps):
            self.env[:, :, :-1] *= self.pc
            prey_locations = self.process_preys(time)
            
            if self.prey_counter == 0:
                break

            if self.log_env:
                self.env_log.append(np.copy(self.env))

            location, reward = self.agnt.training_act(self.env_log[-1], prey_locations)
            training_trial_data['path'].append(location)
            training_trial_data['rewards'].append(reward)
            training_trial_data['prey_locations'].append(prey_locations)
        training_trial_data['env'] = self.env_log

        return training_trial_data
    
    def run_trial(self):
        test_trial_data = {'path': [], 'preys': [], 'prey_locations': [], 'env': []}

        for time in range(self.n_steps):
            self.env[:, :, :-1] *= self.pc
            prey_locations = self.process_preys(time)
            
            if self.prey_counter == 0:
                break

            if self.log_env:
                self.env_log.append(np.copy(self.env))

            location = self.agnt.act(self.env_log[-1], prey_locations)
            test_trial_data['path'].append(location)
            test_trial_data['prey_locations'].append(prey_locations)
        test_trial_data['prey'] = self.preys
        test_trial_data['env'] = self.env_log

        return test_trial_data

    def process_preys(self, time):
        prey_locations = []

        for prey in self.preys:
            if prey.state == 1:
                prey_locations.append(prey.location.copy())
                if np.array_equal(prey.location, self.agnt.location):
                    prey.state = 0
                    self.prey_counter -= 1

                    if self.scenario == 'Two Prey':
                        self.prey_counter = 0
                else:
                    if self.scenario != "Static" and np.random.rand() < self.pm:
                        prey.move(self.env)
                    if prey.collision == 1:
                        self.prey_counter -= 1
                        continue
                    
                    prey.path.append(prey.location.copy())
                    self.emit_cues(prey, time)
                    # self.emit_noise(prey)

        return np.array(prey_locations)

    def emit_cues(self, prey, time):
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
        self.pk_hw = self.pk // 2
        r, c = prey.location
        cue_top = r - self.pk_hw // 8
        cue_bottom = r + self.pk_hw // 8
        cue_left = c - self.pk_hw // 8
        cue_right = c + self.pk_hw // 8

        np.random.shuffle(self.k2d_noise.reshape(-1))
        self.env[cue_top: cue_bottom, cue_left: cue_right, prey.cues[1]] += self.k2d_noise[:cue_bottom - cue_top, :cue_right - cue_left]


class LinearPreyEvaluator:
    def __init__(self, width, height, agnt, sensor_noise_scale, n_prey, pk, n_steps, scenario, case, motion, visible_steps, multisensory, pc, pm=None, pe=None):
        self.width = width
        self.height = height
        self.agnt = agnt
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

    def train_RL(self, training_trials):
        self.training_trials = {}
        self.trial_lengths = []

        for trial in tqdm(range(training_trials)):
            if self.scenario == 'Constant':
                self.case = str(np.random.randint(1, 4))
                self.pm = np.random.rand()
            self.training_trial = PredatorTrial(width=self.width, height=self.height, agnt=self.agnt, sensor_noise_scale=self.sensor_noise_scale, n_prey=self.n_prey, pk=self.pk, n_steps=self.n_steps, scenario=self.scenario, case=self.case, motion=self.motion, visible_steps=self.visible_steps, multisensory=self.multisensory, pc=self.pc, pm=self.pm, pe=self.pe)
            training_trial_data = self.training_trial.run_training_trial()
            self.training_trials[trial] = training_trial_data
            self.training_trials[trial]['prey_states'] = [prey.state for prey in self.training_trial.preys]
            self.trial_lengths.append(len(training_trial_data['path']))

    def training_plots(self, training_lengths=False, first_5_last_5=False, percentage_captured=False, animate=[False, None]):
        self.agnt.produce_plots(training_lengths=training_lengths, first_5_last_5=first_5_last_5, percentage_captured=percentage_captured, animate=animate, trials=self.training_trials, trial_lengths=self.trial_lengths)

    def evaluate(self, n_trials, case, pm):
        self.test_trials = {}
        
        for trial in range(n_trials):
            test_trial = PredatorTrial(width=self.width, height=self.height, agnt=self.agnt, sensor_noise_scale=self.sensor_noise_scale, n_prey=self.n_prey, pk=self.pk, n_steps=self.n_steps, scenario=self.scenario, case=case, motion=self.motion, visible_steps=self.visible_steps, multisensory=self.multisensory, pc=self.pc, pm=pm, pe=self.pe)
            test_trial_data = test_trial.run_trial()
            self.test_trials[trial] = test_trial_data
            
        captured, approached = self.calculate_success(self.test_trials, n_trials)
        return self.test_trials, captured, approached

    def calculate_success(self, trials, n_trials):
        total = 0
        captured = 0
        approached = 0
        
        for trial in range(n_trials):
            # prey_states = [prey.state for prey in trials[trial]['prey']]
            # print(prey_states)
            # total += len(prey_states)
            # captured += len(prey_states) - sum(prey_states)
            prey_captured = [prey.location.copy() for prey in trials[trial]['prey'] if prey.state == 0]
            prey_not_captured = [prey.location.copy() for prey in trials[trial]['prey'] if prey.state == 1]
            last_agent_location = trials[trial]['path'][-1]
            
            for prey_location in prey_not_captured:
                if (last_agent_location == prey_location + np.array([1, 0])).all() or (last_agent_location == prey_location + np.array([0, 1])).all() or (last_agent_location == prey_location + np.array([0, -1])).all():
                    approached += 1

            total += len(prey_captured) + len(prey_not_captured)
            captured += len(prey_captured)
            approached += len(prey_captured)
        captured = 100 * captured / total
        approached = 100 * approached / total

        return captured, approached