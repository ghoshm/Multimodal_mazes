# Maze experiment

import numpy as np
import neat

from agent import *
from maze_env import *


def eval_fitness(genome, config, n_steps=50):
    """
    Evalutes the fitness of the provided genome.
    Arguments:
        genome: neat generated genome.
        config: the neat configuration holder.
        n_steps: number of simulation steps.
    Returns:
        fitness: the mean fitness across mazes, between [0,1].
    """
    fitness = []
    fastest_solution = ((track.size - 2) // 2) - 1
    # For each maze
    for mz_n, mz in enumerate(track.mazes):
        agnt = agent(location=[5, 5], channels=[1, 1], genome=genome, config=config)

        # Sensation-action loop
        for t in range(n_steps):
            agnt.sense(mz)
            agnt.act(mz)

            # If the end is reached
            if agnt.location[1] == track.goal_locations[mz_n]:
                break

        # Record fitness
        fitness.append(t)

    # ADD: store simple info on each agent (see book, maze_exp.py line 72-83)

    # Normalise fitness
    fitness = 1 - (
        (np.array(fitness).mean() - fastest_solution) / (n_steps - 1 - fastest_solution)
    )

    # Return fitness
    return fitness


def eval_genomes(genomes, config):
    """
    Evaluates the fitness of each genome in the population.
    Arguments:
        genomes: the list of genomes in the current population.
        config: the neat configuration holder.
    Updates:
        genome.fitness
    """
    for _, genome in genomes:
        genome.fitness = eval_fitness(genome, config)


def run(config_file, n_generations):
    """
    Runs the experiment.
    Arguments:
        config_file: the path to the config.ini file.
        n_generations: the number of generations to run.
    """
    # Load config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Create the population
    p = neat.Population(config)

    # Reporting
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run (needs to call eval_genomes)
    best_genome = p.run(eval_genomes, n=n_generations)

    # Output
    print(best_genome.fitness)
    print(best_genome.size())
    print(best_genome.connections)


if __name__ == "__main__":
    # Add user arguments (with argparse)
    n_generations = 50

    # Generate mazes
    track = maze(size=11, n_channels=2)
    track.generate_track_mazes(50)

    # Run
    run("./maze_config.ini", n_generations)
