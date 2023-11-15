# Maze experiment

import numpy as np
import neat
import pickle
import argparse

import multimodal_mazes


def eval_fitness(genome_id, genome, config, n_steps=50):
    """
    Evalutes the fitness of the provided genome.
    Arguments:
        genome_id: neat generated genome number.
        genome: neat generated genome.
        config: the neat configuration holder.
    Returns:
        fitness: the mean fitness across mazes, between [0,1].
    """
    if genome_id == 1:
        global max_fitness, top_genome
        max_fitness = 0.0
        top_genome = []

    fitness = []
    # For each maze
    for mz_n, mz in enumerate(maze.mazes):
        # Run trial
        time, _ = multimodal_mazes.maze_trial(
            mz,
            maze.goal_locations[mz_n],
            args.channels,
            genome,
            config,
            n_steps,
        )

        # Record fitness
        fitness.append(time)

    # Normalise fitness
    fitness = 1 - (
        (np.array(fitness) - maze.fastest_solutions)
        / (n_steps - 1 - maze.fastest_solutions)
    )

    # # Calculate fitness per channel
    # fitness = [
    #     fitness[maze.goal_channels == ch].mean() for ch in np.unique(maze.goal_channels)
    # ]

    # Record data
    agent_record.append(
        (
            genome_id,
            p.generation,
            p.species.get_species_id(genome_id),
            np.array(fitness).mean(),
            # fitness[0],
            # fitness[1],
        )
    )

    # Track top genome
    # Could do this per species
    if np.array(fitness).mean() > max_fitness:
        max_fitness = np.array(fitness).mean()
        top_genome = [genome_id, genome, args.channels]

    # Return fitness
    return np.array(fitness).mean()


def eval_genomes(genomes, config):
    """
    Evaluates the fitness of each genome in the population.
    Arguments:
        genomes: the list of genomes in the current population.
        config: the neat configuration holder.
    Updates:
        genome.fitness
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_fitness(genome_id, genome, config)


def run_exp(config_path, n_generations):
    """
    Runs the experiment.
    Arguments:
        config_path: the path to the config.ini file.
        n_generations: the number of generations to run.
    """
    # Load config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create the population
    global p  # encourage not to use global variables
    p = neat.Population(config)

    # Reporting
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run
    final_genome = p.run(eval_genomes, n=n_generations)


if __name__ == "__main__":
    # User hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        help="specify active (1) and inative (0) channels e.g. 0 1.",
    )
    parser.add_argument(
        "--maze_size",
        type=int,
        help="the size of the square maze.",
    )
    parser.add_argument(
        "--n_mazes",
        type=int,
        help="the number of mazes to generate.",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        help="the number of generations to run.",
    )
    args = parser.parse_args()

    # Other hyperparameters
    config_path = "../neat_config.ini"

    # Generate mazes
    maze = multimodal_mazes.HMaze(size=args.maze_size, n_channels=len(args.channels))
    maze.generate(args.n_mazes)

    # Run
    agent_record = []
    run_exp(config_path, args.n_generations)

    # Save results
    agent_record = np.array(
        agent_record,
        dtype=[
            ("genome_id", "uint64"),
            ("generation", "uint64"),
            ("species", "uint64"),
            ("fitness", "float64"),
            # ("ch0_fitness", "float64"),
            # ("ch1_fitness", "float64"),
        ],
    )
    np.save("../results/test.npy", agent_record)
    with open("../results/test.pickle", "wb") as file:
        pickle.dump(top_genome, file)
