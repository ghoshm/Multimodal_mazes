# robustness
import numpy as np
import multimodal_mazes
from tqdm import tqdm


def robustness_to_maze_noise(
    agents, noise_scales, n_mazes, exp_config, genomes, config
):
    """
    Tests agents robustness to noise applied to the maze.
    Arguments:
        agents: a list of indicies, of the genomes to test.
        noise_scales: a np vector of noise scales to use.
        n_mazes: the number of mazes to test per noise_scale.
        exp_config: a dict with the exp configuration.
        genomes: neat generated genomes.
        config: the neat configuration holder.
    Returns:
        noise_results: a np array with each agents fitness per noise_scale.
        noise_baseline: as noise_results, but using random agents.
    """
    noise_results = np.zeros((len(agents), len(noise_scales)))

    agnt_random_baseline = multimodal_mazes.AgentRandom(
        location=None,
        channels=[0, 0],
    )
    noise_baseline = np.zeros((len(agents), len(noise_scales)))

    for a, noise_scale in enumerate(tqdm(noise_scales)):
        # Generate mazes
        maze = multimodal_mazes.TrackMaze(
            size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
        )
        maze.generate(number=n_mazes, noise_scale=noise_scale)

        # Test agents
        for b, g in enumerate(agents):
            _, genome, channels = genomes[g]
            genome = multimodal_mazes.prune_architecture(genome, config)
            noise_results[b, a] = multimodal_mazes.eval_fitness(
                genome=genome,
                config=config,
                channels=channels,
                sensor_noise_scale=exp_config["sensor_noise_scale"],
                drop_connect_p=exp_config["drop_connect_p"],
                maze=maze,
                n_steps=exp_config["n_steps"],
            )

            # Random baseline
            noise_baseline[b, a] = multimodal_mazes.eval_fitness(
                genome=None,
                config=None,
                channels=[0, 0],
                sensor_noise_scale=0.0,
                drop_connect_p=0.0,
                maze=maze,
                n_steps=exp_config["n_steps"],
                agnt=agnt_random_baseline,
            )

    return noise_results, noise_baseline


def robustness_to_sensor_noise(
    agents, noise_scales, n_mazes, exp_config, genomes, config
):
    """
    Tests agents robustness to noise applied to their input sensors.
    Arguments:
        agents: a list of indicies, of the genomes to test.
        noise_scales: a np vector of noise scales to use.
        n_mazes: the number of mazes to test per noise_scale.
        exp_config: a dict with the exp configuration.
        genomes: neat generated genomes.
        config: the neat configuration holder.
    Returns:
        noise_results: a np array with each agents fitness per noise_scale.
        noise_baseline: as noise_results, but using random agents.
    """
    noise_results = np.zeros((len(agents), len(noise_scales)))
    noise_baseline = np.zeros((len(agents), len(noise_scales)))

    # Generate mazes
    maze = multimodal_mazes.TrackMaze(
        size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
    )
    maze.generate(
        number=n_mazes,
        noise_scale=exp_config["maze_noise_scale"],
        gaps=exp_config["maze_gaps"],
    )

    for a, noise_scale in enumerate(tqdm(noise_scales)):
        agnt_random_baseline = multimodal_mazes.AgentRandom(
            location=None, channels=[0, 0]
        )

        # Test agents
        for b, g in enumerate(agents):
            _, genome, channels = genomes[g]
            genome = multimodal_mazes.prune_architecture(genome, config)
            noise_results[b, a] = multimodal_mazes.eval_fitness(
                genome=genome,
                config=config,
                channels=channels,
                sensor_noise_scale=noise_scale,
                drop_connect_p=exp_config["drop_connect_p"],
                maze=maze,
                n_steps=exp_config["n_steps"],
            )

            # Random baseline
            noise_baseline[b, a] = multimodal_mazes.eval_fitness(
                genome=None,
                config=None,
                channels=[0, 0],
                sensor_noise_scale=0.0,
                drop_connect_p=0.0,
                maze=maze,
                n_steps=exp_config["n_steps"],
                agnt=agnt_random_baseline,
            )

    return noise_results, noise_baseline


def robustness_to_drop_connect(
    agents, drop_connect_ps, n_mazes, exp_config, genomes, config
):
    """
    Tests agents robustness to dropping connections.
    Arguments:
        agents: a list of indicies, of the genomes to test.
        drop_connect_ps: a np vector of drop levels to use.
        n_mazes: the number of mazes to test per drop level.
        exp_config: a dict with the exp configuration.
        genomes: neat generated genomes.
        config: the neat configuration holder.
    Returns:
        drop_results: a np array with each agents fitness per drop level.
        drop_baseline: as drop_results, but using random agents.
    """
    drop_results = np.zeros((len(agents), len(drop_connect_ps)))
    drop_baseline = np.zeros((len(agents), len(drop_connect_ps)))

    # Generate mazes
    maze = multimodal_mazes.TrackMaze(
        size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
    )
    maze.generate(
        number=n_mazes,
        noise_scale=exp_config["maze_noise_scale"],
        gaps=exp_config["maze_gaps"],
    )

    for a, drop_connect_p in enumerate(tqdm(drop_connect_ps)):
        agnt_random_baseline = multimodal_mazes.AgentRandom(
            location=None, channels=[0, 0]
        )

        # Test agents
        for b, g in enumerate(agents):
            _, genome, channels = genomes[g]
            genome = multimodal_mazes.prune_architecture(genome, config)
            drop_results[b, a] = multimodal_mazes.eval_fitness(
                genome=genome,
                config=config,
                channels=channels,
                sensor_noise_scale=exp_config["sensor_noise_scale"],
                drop_connect_p=drop_connect_p,
                maze=maze,
                n_steps=exp_config["n_steps"],
            )

            # Random baseline
            drop_baseline[b, a] = multimodal_mazes.eval_fitness(
                genome=None,
                config=None,
                channels=[0, 0],
                sensor_noise_scale=0.0,
                drop_connect_p=0.0,
                maze=maze,
                n_steps=exp_config["n_steps"],
                agnt=agnt_random_baseline,
            )

    return drop_results, drop_baseline
