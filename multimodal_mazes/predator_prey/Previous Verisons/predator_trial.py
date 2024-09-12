# Predator trial

import numpy as np
import multimodal_mazes
from scipy import signal


def predator_trial(
    size,
    agnt,
    sensor_noise_scale,
    n_prey,
    pk,
    n_steps,
    scenario,
    motion,
    pc,
    pm=None,
    pe=None,
    log_env=False,
):
    """
    Tests a single agent on a single predator trial.
    Arguments:
        size: the size of the square environment.
        agnt: a network or algorithm which implements a policy function.
        sensor_noise_scale: added to each sensor at each time step.
        n_prey: the number of prey which start in the environment.
        pk: the width of the prey's Gaussian signal (in rc).
        n_steps: the number of steps the simulation lasts.
        scenario: defines the task as either foraging or hunting.
        motion: the type of motion used by the prey, either Brownian or Levy.
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
    env = np.zeros((size, size, len(agnt.channels) + 1))
    env[:, :, -1] = 1.0
    env = np.pad(env, pad_width=((pk_hw, pk_hw), (pk_hw, pk_hw), (0, 0)))
    env_log = [np.copy(env)]

    # Reset agent
    agnt.location = np.array([pk_hw + (size // 2), pk_hw + (size // 2)])
    agnt.sensor_noise_scale = sensor_noise_scale
    agnt.outputs *= 0.0
    if agnt.type == "Hidden skip":
        agnt.memory = np.zeros_like(agnt.outputs)
    elif agnt.type == "Levy":
        agnt.flight_length = 0
        agnt.collision = 0

    # Define prey
    k1d = signal.windows.gaussian(pk, std=1)
    k2d = np.outer(k1d, k1d)
    k2d_noise = np.copy(k2d)

    rcs = np.stack(np.argwhere(env[:, :, -1]))
    prey_rcs = np.random.choice(range(len(rcs)), size=n_prey, replace=False)
    preys = []
    
    for n in range(n_prey):
        preys.append(
            multimodal_mazes.AgentRandom(
                location=rcs[prey_rcs[n]], channels=[0, 0], motion=motion
            )
        )
        preys[n].state = 1  # free (1) or caught (0)
        preys[n].path = [list(preys[n].location)]

        if scenario == "Foraging":
            preys[n].cues = n % 2  # channel for emitting cues

    # Sensation-action loop
    path = [list(agnt.location)]
    prey_counter = np.copy(n_prey)
    for time in range(n_steps):

        env[:, :, :-1] *= pc  # reset channels

        # Prey
        for prey in preys:
            if prey.state == 1:
                if (prey.location == agnt.location).all():  # caught
                    prey.state = 0
                    prey_counter -= 1

                else:  # free
                    # Movement
                    if scenario == "Hunting":
                        if np.random.rand() < pm:
                            prey.policy()
                            prey.act(env)
                    prey.path.append(list(prey.location))

                    # Emit cues
                    r, c = prey.location
                    if scenario == "Foraging":
                        env[
                            r - pk_hw : r + pk_hw + 1,
                            c - pk_hw : c + pk_hw + 1,
                            prey.cues,
                        ] += np.copy(k2d)

                    elif scenario == "Hunting":
                        for ch in range(len(prey.channels)):
                            if np.random.rand() < pe:
                                env[
                                    r - pk_hw : r + pk_hw + 1,
                                    c - pk_hw : c + pk_hw + 1,
                                    ch,
                                ] += np.copy(
                                    k2d
                                )  # emit cues
                            else:
                                np.random.shuffle(k2d_noise.reshape(-1))
                                env[
                                    r - pk_hw : r + pk_hw + 1,
                                    c - pk_hw : c + pk_hw + 1,
                                    ch,
                                ] += np.copy(
                                    k2d_noise
                                )  # emit noise

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


def eval_predator_fitness(
    n_trials,
    size,
    agnt,
    sensor_noise_scale,
    n_prey,
    pk,
    n_steps,
    scenario,
    motion,
    pc,
    pm=None,
    pe=None,
):
    """
    Evaluates the fitness of an agent across multiple predator trials.
    Arguments:
        n_trials: the number of trials to run.
        size: the size of the square environment.
        agnt: a network or algorithm which implements a policy function.
        sensor_noise_scale: added to each sensor at each time step.
        n_prey: the number of prey which start in the environment.
        pk: the width of the prey's Gaussian signal (in rc).
        n_steps: the number of steps the simulation lasts.
        scenario: defines the task as either foraging or hunting.
        motion: the type of motion used by the prey, either Brownian or Levy.
        pc: the persistence of cues in the environment from 0 to 1 (instantaneous to constant).
        pm: the probability of prey moving (per timestep) when hunting.
        pe: the probability of prey emitting cues (per timestep) when hunting.
    Returns:
        fitness: the mean fitness across trials, between [0,1].
        times: a np vector with the length of each trial.
        paths: a list of np arrays with the predators' path per trial [r,c].
        preys: a list of lists containing the prey agents.
    """
    fitness, times, paths, preys = [], [], [], []

    # For each trial
    for _ in range(n_trials):
        # Run trial
        time, path, prey_state, prey, _ = predator_trial(
            size=size,
            agnt=agnt,
            sensor_noise_scale=sensor_noise_scale,
            n_prey=n_prey,
            pk=pk,
            n_steps=n_steps,
            scenario=scenario,
            motion=motion,
            pc=pc,
            pm=pm,
            pe=pe,
        )

        fitness.append(prey_state)
        times.append(time)
        paths.append(path)
        preys.append(prey)

    fitness = np.array(fitness)

    return (
        (1 - fitness).sum() / fitness.size,
        np.array(times),
        paths,
        preys,
    )


def prey_params_search(
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
