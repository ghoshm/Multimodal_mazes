# Maze experiment

import numpy as np
import neat

from agent import *
from maze_env import *


def eval_fitness(genome, config, n_steps=50):
    """ """
    agnt = agent(location=[4, 4], channels=[0, 1], genome=genome, config=config)

    fitness = []
    # For each maze
    for mz_n, mz in enumerate(track.mazes):
        # Reset the agents location
        agnt.location = [4, 4]

        # Sensation-action loop
        for t in range(n_steps):
            agnt.sense(mz)
            agnt.act(mz)

            # If the end is reached
            if agnt.location[1] == track.goals[mz_n]:
                break

        # Record fitness
        fitness.append(t)

    # Normalise fitness
    fitness = 1 - ((np.array(fitness).mean() - 3) / 46)

    # Return fitness
    return fitness


def eval_genomes(genomes, config):
    """ """
    for _, genome in genomes:
        genome.fitness = eval_fitness(genome, config)


def run(config_file, n_generations):
    """ """
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


if __name__ == "__main__":
    # Add user arguments (with argparse)
    n_generations = 50

    # Generate mazes
    track = maze(size=9, n_channels=2, goal_grad=0, avoid_grad=0)
    track.generate_track_mazes(50)

    # Run
    run("./maze_config.ini", n_generations)
