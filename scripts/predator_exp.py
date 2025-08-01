# Predator experiment

import numpy as np
import neat
import pickle
import os
import shutil

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
        genome.fitness, _, _, _ = multimodal_mazes.eval_predator_fitness(
            n_trials=exp_config["n_trials"],
            size=exp_config["size"],
            agnt=None,
            sensor_noise_scale=exp_config["sensor_noise_scale"],
            n_prey=exp_config["n_prey"],
            pk=exp_config["pk"],
            n_steps=exp_config["n_steps"],
            scenario=exp_config["scenario"],
            motion=exp_config["motion"],
            pc=exp_config["pc"],
            pm=exp_config["pm"],
            pe=exp_config["pe"],
            channels=exp_config["channels"],
            drop_connect_p=exp_config["drop_connect_p"],
            genome=genome,
            config=config,
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
        genome_record.append([genome_id, genome, exp_config["channels"]])


def run_exp(neat_config_path, n_generations):
    """
    Runs the experiment.
    Arguments:
        neat_config_path: the path to the neat_config.ini file.
        n_generations: the number of generations to run.
    """
    # Load config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )

    # Create the population
    global p  # encourage not to use global variables
    p = neat.Population(config)

    # Init with positive weights
    for n in p.population:
        for c in p.population[n].connections:
            p.population[n].connections[c].weight = abs(
                p.population[n].connections[c].weight
            )

    # Reporting
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run
    _ = p.run(eval_genomes, n=n_generations)


if __name__ == "__main__":
    # Config files
    neat_config_path = "../neat_config.ini"
    exp_config = multimodal_mazes.load_prey_config("../prey_config.ini")

    # Create save folder
    os.makedirs(exp_config["save_path"], exist_ok=True)

    # Copy config files to save folder
    shutil.copyfile("../neat_config.ini", exp_config["save_path"] + "/neat_config.ini")
    shutil.copyfile("../prey_config.ini", exp_config["save_path"] + "/prey_config.ini")

    # Run
    agent_record, genome_record = [], []
    run_exp(neat_config_path, exp_config["n_generations"])

    # Save results
    try:
        job_index = str(int(os.environ["PBS_ARRAY_INDEX"]) - 1)  # array job
    except:
        job_index = str(99)  # single job
    save_path = exp_config["save_path"] + "/" + job_index

    agent_record = np.array(
        agent_record,
        dtype=[
            ("genome_id", "uint64"),
            ("generation", "uint64"),
            ("species", "uint64"),
            ("fitness", "float64"),
        ],
    )
    np.save(save_path, agent_record)
    with open(save_path + ".pickle", "wb") as file:
        pickle.dump(genome_record, file)
