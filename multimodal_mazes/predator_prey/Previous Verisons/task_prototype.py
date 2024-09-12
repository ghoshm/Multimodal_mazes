# Predator trial
import matplotlib.pyplot as plt
import numpy as np
import multimodal_mazes
from scipy import signal

def linear_prey_trial(
    width,
    height,
    agnt,
    sensor_noise_scale,
    n_prey,
    pk,
    n_steps,
    scenario,
    case,
    motion,
    visible_steps,
    multisensory,
    pc,
    pm=None,
    pe=None,
    log_env=False,
):
    """
    Tests a single agent on a single predator trial.
    Arguments:
        width: the width of the rectangular environment.
        height: the height of the rectangular environment.
        agnt: a network or algorithm which implements a policy function.
        sensor_noise_scale: added to each sensor at each time step.
        n_prey: the number of prey which start in the environment.
        pk: the width of the prey's Gaussian signal (in rc).
        n_steps: the number of steps the simulation lasts.
        scenario: defines the task as either foraging or hunting.
        case: determines the start position of the prey, cand be 1, 2, 3.
        motion: the type of motion used by the prey, linear or None.
        visible_stpes: the number of time steps for which the prey should be visible to the agent.
        multisensory: determines whether additional cues will be emitted.
        pc: the persistence of cues in the environment from 0 to 1 (instantaneous to constant).
        pm: the probability of prey moving (per timestep) when hunting.
        pe: the probability of prey emitting cues (per timestep) when hunting.
        log_env: record the env state (True) or not (False).
    Returns:
        time: the number of steps taken to catch all prey.
              Returns n_steps-1 if the agent fails.
        path: a np array with the agent's location at each time step [r,c].
        prey state: a list with the final state (0 or 1, caught or free) of each prey.
        preys: a list containing the prey agents.
        env_log: a list storing the env at every step.
    """
    pk_hw = pk // 2  # half width of prey's Gaussian signal (in rc)

    # Create environment with track (1. and walls 0.)
    env = np.zeros((height, width, len(agnt.channels) + 1))
    env[:, :, -1] = 1.0
    env = np.pad(env, pad_width=((pk_hw, pk_hw), (pk_hw, pk_hw), (0, 0)))
    env_log = [np.copy(env)]

    # Reset agent
    agnt.location = np.array([pk_hw + (height // 2), pk_hw + (width // 2)])
    agnt.sensor_noise_scale = sensor_noise_scale
    agnt.outputs *= 0.0
    if agnt.type == "Hidden skip":
        agnt.memory = np.zeros_like(agnt.outputs)
    elif agnt.type == "Levy":
        agnt.flight_length = 0
        agnt.collision = 0

    # Define prey
    k1d = signal.windows.gaussian(pk, std=9)
    k2d = np.outer(k1d, k1d)
    k1d_noise = signal.windows.gaussian(pk//8, std=1)
    k2d_noise = np.outer(k1d_noise, k1d_noise)
    
    # Define possible directions
    preys = []
    directions = [-1, 1]
    direction = 0

    # Define start positions for the different cases
    if scenario == 'Static':
        rcs = np.stack(np.argwhere(env[:, :, -1]))
        prey_rcs = np.random.choice(range(width), size=n_prey, replace=False)
    elif scenario != 'Static':
        possible_starts = [[[pk_hw, pk_hw+width//2]], [[pk_hw, width+pk_hw-1], [pk_hw, pk_hw]], [[pk_hw, pk_hw+(width//4)], [pk_hw, pk_hw+((3*width)//4)]], [[pk_hw, width+pk_hw-5], [pk_hw, pk_hw+4]]]
        choice = np.random.choice(range(2)) 
        if case == '4':
            direction = [directions[choice], directions[1-choice]]
            start_rc = [possible_starts[int(case)-1][choice], possible_starts[int(case)-1][1-choice]]
        else:
            direction = [directions[choice]]
            start_rc = [possible_starts[int(case)-1][choice]] if len(possible_starts[int(case)-1]) == 2 else [possible_starts[int(case)-1][0]]

        # if case == "1":
        #     start_rc = [[pk_hw, pk_hw+width//2]]
        #     direction = [directions[choice]]
        # elif case == "2":
        #     possible_starts = [[pk_hw, width+pk_hw-1], [pk_hw, pk_hw]]
        #     start_rc = [possible_starts[choice]]
        #     direction = [directions[choice]]
        # elif case == "3":
        #     possible_starts = [[pk_hw, pk_hw+(width//4)], [pk_hw, pk_hw+((3*width)//4)]]
        #     start_rc = [possible_starts[choice]]
        #     direction = [directions[choice]]
        # elif case == "4":
        #     #possible_starts = [[pk_hw, pk_hw+(width//2)-2], [pk_hw, pk_hw+(width//2)+2]]
        #     possible_starts = [[pk_hw, width+pk_hw-5], [pk_hw, pk_hw+4]]
        #     start_rc = [possible_starts[choice], possible_starts[1-choice]]
        #     direction = [directions[choice], directions[1-choice]]
        
    for n in range(n_prey):
        if scenario == "Static":
            preys.append(
                multimodal_mazes.PreyLinear(
                    location=rcs[prey_rcs[n]], channels=[0, 0], scenario=scenario, motion=motion, direction=direction
                )
            )
        elif scenario != "Static":
            preys.append(
                multimodal_mazes.PreyLinear(
                    location=start_rc[n], channels=[0, 0], scenario=scenario, motion=motion, direction=direction[n]
                )
            )
        preys[n].state = 1  # free (1) or caught (0)
        preys[n].path = [list(preys[n].location)]

        if scenario != "Two Prey":
            preys[n].cues = (n % 2, ((n+1) % 2))
        else:
            if n == 0:
                cue = np.random.choice(range(2))
                preys[n].cues = (cue, 1-cue)
            preys[n].cues = (1-cue, cue)
    
    
    # Sensation-action loop
    path = [list(agnt.location)]
    prey_counter = np.copy(n_prey)
    
    for time in range(n_steps):

        env[:, :, :-1] *= pc  # reset channels

        # Prey
        for n, prey in enumerate(preys):
            if prey.state == 1:
                if (prey.location == agnt.location).all():  # caught
                    prey.state = 0
                    prey_counter -= 1
                    if scenario == "Two Prey":
                        prey_counter = 0

                else: # free
                    if scenario == "Two Prey":
                        if n == 0:
                            random_num = np.random.rand()
                        if random_num < pm:
                            prey.move(env)
                    if scenario != "Static" and np.random.rand() < pm:
                    #if scenario != "Static" and (int(str(time)[-1]) / 10 <= pm):
                        prey.move(env)
                        
                    if prey.collision == 1:  # trial failed
                        prey_counter -= 1     
                    else:  # trial not failed
                        prey.path.append(list(prey.location))

                        # Emit cues
                        r, c = prey.location                        
                        
                        if np.random.rand() < pe:
                            cue_top = r - pk_hw
                            cue_bottom = r + pk_hw
                            cue_left = c - pk_hw
                            cue_right = c + pk_hw

                            if scenario == "Static":
                                env[
                                    cue_top: cue_bottom,
                                    cue_left : cue_right,
                                    prey.cues[0],
                                ] += k2d[:cue_bottom - cue_top, :cue_right - cue_left]
                            elif scenario != "Static" and time <= visible_steps:
                                if multisensory == "Balanced" and n == 0:
                                    env[
                                        cue_top: cue_bottom,
                                        cue_left : cue_right,
                                        prey.cues[0],
                                    ] += (0.5*k2d[:cue_bottom - cue_top, :cue_right - cue_left])
                                    env[
                                        cue_top: cue_bottom,
                                        cue_left : cue_right,
                                        prey.cues[1],
                                    ] += (0.5*k2d[:cue_bottom - cue_top, :cue_right - cue_left])
                                else:
                                    env[
                                        cue_top: cue_bottom,
                                        cue_left : cue_right,
                                        prey.cues[0],
                                    ] += (k2d[:cue_bottom - cue_top, :cue_right - cue_left])
                                    
                        else:
                            cue_top = r - pk_hw//8
                            cue_bottom = r + pk_hw//8
                            cue_left = c - pk_hw//8
                            cue_right = c + pk_hw//8

                            np.random.shuffle(k2d_noise.reshape(-1))
                            env[
                                cue_top: cue_bottom,
                                cue_left : cue_right,
                                prey.cues[1],
                            ] += k2d_noise[:cue_bottom - cue_top, :cue_right - cue_left] # emit noise

            if multisensory == "Broad":
                #Define dimensions of broad cue
                ek1dc = signal.windows.boxcar(3*width//4 + 1)
                ek1dr = signal.windows.boxcar(height)
                ek2d = np.outer(ek1dr, ek1dc)

                #Emit broad cue
                bcue_top = pk_hw
                bcue_bottom =  min(pk_hw + height, pk_hw + height)
                bcue_left = max(pk_hw, prey.location[1] - (3 * width // 8))
                bcue_right = min(prey.location[1] + (3 * width // 8), pk_hw + width)

                if time <= visible_steps:
                    env[
                        bcue_top:bcue_bottom,
                        bcue_left:bcue_right,
                        prey.cues[1],
                    ] += ek2d[:bcue_bottom - bcue_top, :bcue_right - bcue_left]

        # Apply edges
        for ch in range(len(agnt.channels)):
            env[:, :, ch] *= env[:, :, -1]

        # If all prey have been caught
        if prey_counter == 0:
            break

        # Log env
        if log_env:
            env_log.append(np.copy(env))

        # Predator
        agnt.sense(env)
        agnt.policy()
        agnt.act(env)

        path.append(list(agnt.location))

    return (
        time,
        np.array(path),
        [preys[n].state for n in range(n_prey)],
        preys,
        env_log,
        )


def eval_linear_prey_fitness(
    n_trials,
    width, 
    height,
    agnt,
    sensor_noise_scale,
    n_prey,
    pk,
    n_steps,
    scenario,
    case,
    motion,
    visible_steps,
    multisensory,
    pc,
    pm=None,
    pe=None,
):
    """
    Evaluates the fitness of an agent across multiple predator trials.
    Arguments:
        n_trials: the number of trials to run.
        width: the width of the rectangular environment.
        height: the height of the rectangular environment.agnt: a network or algorithm which implements a policy function.
        sensor_noise_scale: added to each sensor at each time step.
        n_prey: the number of prey which start in the environment.
        pk: the width of the prey's Gaussian signal (in rc).
        n_steps: the number of steps the simulation lasts.
        scenario: defines the task as either foraging or hunting.
        case: determines the start position of the prey, cand be 1, 2, 3.
        motion: the type of motion used by the prey, linear or None.
        visible_stpes: the number of time steps for which the prey should be visible to the agent.
        multisensory: determines whether additional cues will be emitted.
        pc: the persistence of cues in the environment from 0 to 1 (instantaneous to constant).
        pm: the probability of prey moving (per timestep) when hunting.
        pe: the probability of prey emitting cues (per timestep) when hunting.
    Returns:
        fitness: the mean fitness across trials, between [0,1].
        times: a np vector with the length of each trial.
        paths: a list of np arrays with the predators' path per trial [r,c].
        preys: a list of lists containing the prey agents.
        captured: the percentage of prey captured.
    """
    fitness, times, paths, preys = [], [], [], []

    # For each trial
    for _ in range(n_trials):
        # Run trial
        time, path, prey_state, prey, _ = linear_prey_trial(
            width=width,
            height=height,
            agnt=agnt,
            sensor_noise_scale=sensor_noise_scale,
            n_prey=n_prey,
            pk=pk,
            n_steps=n_steps,
            scenario=scenario,
            case=case,
            motion=motion,
            visible_steps=visible_steps,
            multisensory=multisensory,
            pc=pc,
            pm=pm,
            pe=pe,
        )

        fitness.append(prey_state)
        times.append(time)
        paths.append(path)
        preys.append(prey)

    captured = 0
    for prey in preys:
        for n in range(len(prey)):
            if prey[n].state == 0:
                captured += 1

    captured = (captured / n_trials) * 100

    fitness = np.array(fitness)

    return (
        (1 - fitness).sum() / fitness.size,
        np.array(times),
        paths,
        preys,
        captured
    )


def linear_prey_params_search(
    grid_size, size, n_prey, n_steps, n_trials, pk, scenario, motion
):
    """
    Tests rule-based agents over a grid of task parameters.
    Arguments:
        grid_size: the number of values to test per parameter.
        size: the size of the square environment.
        n_prey: the number of prey which start in the environment.
        n_steps: the number of steps the simulation lasts.
        n_trials: the number of trials to run.
        pk: the width of the prey's Gaussian signal (in rc).
        scenario: defines the task as either foraging or hunting.
        motion: the type of motion used by the prey, either Brownian or Levy.
    Returns:
        results: a np array with the fitness results (noises, policies, pms, pes).
        parameters: a dictionary which stores the tested parameters and policies.
    """

    # Set up
    noises = np.linspace(start=0.0, stop=2.0, num=grid_size)
    policies = (
        multimodal_mazes.AgentRuleBased.policies
        + multimodal_mazes.AgentRuleBasedMemory.policies
        + ["Levy"]
    )
    colors = (
        multimodal_mazes.AgentRuleBased.colors
        + multimodal_mazes.AgentRuleBasedMemory.colors
        + [list(np.array([24, 156, 196, 255]) / 255)]
    )
    pms = np.linspace(start=0.0, stop=1.0, num=grid_size)
    pes = np.linspace(start=0.0, stop=1.0, num=grid_size)

    results = np.zeros((len(noises), len(policies), len(pms), len(pes)))

    # Test agents
    for a, noise in enumerate(noises):

        for b, policy in enumerate(policies):
            if policy in multimodal_mazes.AgentRuleBased.policies:
                agnt = multimodal_mazes.AgentRuleBased(
                    location=None, channels=[1, 1], policy=policy
                )
            elif policy in multimodal_mazes.AgentRuleBasedMemory.policies:
                agnt = multimodal_mazes.AgentRuleBasedMemory(
                    location=None, channels=[1, 1], policy=policy
                )
                agnt.alpha = 0.6
            elif policy == "Levy":
                agnt = multimodal_mazes.AgentRandom(
                    location=None, channels=[0, 0], motion=policy
                )

            for c, pm in enumerate(pms):
                for d, pe in enumerate(pes):
                    fitness, _, _, _ = multimodal_mazes.eval_predator_fitness(
                        n_trials=n_trials,
                        size=size,
                        agnt=agnt,
                        sensor_noise_scale=noise,
                        n_prey=n_prey,
                        pk=pk,
                        n_steps=n_steps,
                        scenario=scenario,
                        motion=motion,
                        pc=0.0,  # PLACE HOLDER
                        pm=pm,
                        pe=pe,
                    )

                    results[a, b, c, d] = fitness

    parameters = {
        "noises": noises,
        "policies": policies,
        "colors": colors,
        "pms": pms,
        "pes": pes,
    }

    return results, parameters
