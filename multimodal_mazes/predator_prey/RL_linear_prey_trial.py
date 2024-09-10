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
        self.agnt.location = np.array([self.pk_hw + (self.height // 2), self.pk_hw + (self.width // 2)])
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
            direction = [0 for _ in range(self.n_prey)]
            if self.case == '4':
                direction = [choice, 1-choice]
                start_c = [possible_starts[int(self.case)-1][choice], possible_starts[int(self.case)-1][1-choice]]
            else:
                direction = [choice]
                start_c = [possible_starts[int(self.case)-1][choice]] if len(possible_starts[int(self.case)-1]) == 2 else [possible_starts[int(self.case)-1][0]]

        for n in range(self.n_prey):
            prey = multimodal_mazes.PreyLinear(location=[self.pk_hw, self.pk_hw + start_c[n]], channels=[0, 0], scenario=self.scenario, motion=self.motion, direction=direction[n])
            prey.state = 1
            prey.path = [list(prey.location)]

            if self.scenario != "Two Prey":
                prey.cues = (n % 2, ((n+1) % 2))
            else:
                if n == 0:
                    cue = np.random.choice(range(2))
                    prey.cues = (cue, 1-cue)
                prey.cues = (1-cue, cue)

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

        for time in range(self.n_steps):
            self.env[:, :, :-1] *= self.pc
            prey_locations = self.process_preys(time)
            
            if self.prey_counter == 0:
                break

            if self.log_env:
                self.env_log.append(np.copy(self.env))

            self.agnt.act(self.env_log[-1], prey_locations)
            self.path.append(list(self.agnt.location))

        return time, np.array(self.path), self.preys, self.env_log

    def process_preys(self, time):
        prey_locations = []

        for prey in self.preys:
            if prey.state == 1:
                prey_locations.append(tuple(prey.location))
                if (prey.location == self.agnt.location).all():
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
                    
                    prey.path.append(list(prey.location))
                    self.emit_cues(prey, time)
                    # self.emit_noise(prey)

            if self.multisensory == "Broad":
                self.emit_broad_cue(prey, time)

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

    def emit_broad_cue(self, prey, time):
        self.pk_hw = self.pk // 2
        ek1dc = signal.windows.boxcar(3 * self.width // 4 + 1)
        ek1dr = signal.windows.boxcar(self.height)
        ek2d = np.outer(ek1dr, ek1dc)

        bcue_top = self.pk_hw
        bcue_bottom = min(self.pk_hw + self.height, self.pk_hw + self.height)
        bcue_left = max(self.pk_hw, prey.location[1] - (3 * self.width // 8))
        bcue_right = min(prey.location[1] + (3 * self.width // 8), self.pk_hw + self.width)

        if time <= self.visible_steps:
            self.env[bcue_top: bcue_bottom, bcue_left: bcue_right, prey.cues[1]] += ek2d[:bcue_bottom - bcue_top, :bcue_right - bcue_left]


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

    def train_RL(self):
        self.training_trials = {}
        self.trial_lengths = []

        for trial in tqdm(range(2500)):
            self.training_trial = PredatorTrial(width=self.width, height=self.height, agnt=self.agnt, sensor_noise_scale=self.sensor_noise_scale, n_prey=self.n_prey, pk=self.pk, n_steps=self.n_steps, scenario=self.scenario, case=self.case, motion=self.motion, visible_steps=self.visible_steps, multisensory=self.multisensory, pc=self.pc, pm=self.pm, pe=self.pe)
            training_trial_data = self.training_trial.run_training_trial()
            self.training_trials[trial] = training_trial_data
            self.trial_lengths.append(len(training_trial_data['path']))

    def training_plots(self):
        self.agnt.produce_training_plots(training=True, first_5_last_5=True, training_trials=self.training_trials, trial_lengths=self.trial_lengths)

    def evaluate(self, n_trials):
        times, paths, preys = [], [], []

        for _ in tqdm(range(n_trials)):
            trial = PredatorTrial(width=self.width, height=self.height, agnt=self.agnt, sensor_noise_scale=self.sensor_noise_scale, n_prey=self.n_prey, pk=self.pk, n_steps=self.n_steps, scenario=self.scenario, case=self.case, motion=self.motion, visible_steps=self.visible_steps, multisensory=self.multisensory, pc=self.pc, pm=self.pm, pe=self.pe)
            time, path, prey = trial.run_trial()
            times.append(time)
            paths.append(path)
            preys.append(prey)

        captured, approached = self.calc_success(preys, paths, n_trials)
        return np.array(times), paths, preys, captured, approached

    def calc_success(self, preys, paths, n_trials):
        captured = 0
        approached = 0

        for prey, path in zip(preys, paths):
            for n in range(len(prey)):
                r, c = prey[n].location
                if ((abs(r - path[-1][0]) == 1) and (abs(c - path[-1][1]) == 0)) or ((abs(r - path[-1][0]) == 0) and (abs(c - path[-1][1]) == 1)) or ((abs(r - path[-1][0]) == 1) and (abs(c - path[-1][1]) == 1)):
                    approached += 1
                if prey[n].state == 0:
                    captured += 1
                    approached += 1

        captured = (captured / (n_trials * self.n_prey)) * 100
        approached = (approached / (n_trials * self.n_prey)) * 100
        return captured, approached