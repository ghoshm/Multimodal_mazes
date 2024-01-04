# Maze experiment

import numpy as np
import neat
import pickle
import argparse

import multimodal_mazes


def eval_genomes(genomes, config):
    """
    Evaluates the fitness of each genome in the population.
    Arguments:
        genomes: the list of genomes in the current population.
        config: the neat configuration holder.
    Updates:
        genomes fitness.
        agent_record.
        genome_record.
    """
    for genome_id, genome in genomes:
        genome.fitness = multimodal_mazes.eval_fitness(
            genome=genome, config=config, channels=args.channels, maze=maze
        )

        # Record data
        agent_record.append(
            (
                genome_id,
                p.generation,
                p.species.get_species_id(genome_id),
                genome.fitness,
            )
        )
        genome_record.append([genome_id, genome, args.channels])


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
    _ = p.run(eval_genomes, n=n_generations)


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
    maze = multimodal_mazes.TrackMaze(
        size=args.maze_size, n_channels=len(args.channels)
    )
    maze.generate(args.n_mazes)

    # Run
    agent_record, genome_record = [], []
    run_exp(config_path, args.n_generations)

    # Save results
    agent_record = np.array(
        agent_record,
        dtype=[
            ("genome_id", "uint64"),
            ("generation", "uint64"),
            ("species", "uint64"),
            ("fitness", "float64"),
        ],
    )
    np.save("../results/test.npy", agent_record)
    with open("../results/test.pickle", "wb") as file:
        pickle.dump(genome_record, file)
