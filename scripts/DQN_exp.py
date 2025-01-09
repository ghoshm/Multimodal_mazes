# DQN experiment

import numpy as np
import itertools
import os
import re
import shutil
import pickle
import multimodal_mazes


def run_exp(job_index, exp_config):
    """
    Runs the experiment.
    Arguments:
        job_index: an int which will define the DQN architecture.
        exp_config: loaded hyperparameters.
    Returns:
        agnt: the trained model (and it's test results).
    """
    # Set up
    noises = np.linspace(start=0.0, stop=0.5, num=21)

    wm_flags = np.array(list(itertools.product([0, 1], repeat=7)))
    wm_flags = np.vstack((wm_flags[0], wm_flags))
    wm_flag = wm_flags[job_index]

    # Generate mazes
    maze = multimodal_mazes.TrackMaze(
        size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
    )
    maze.generate(
        number=exp_config["n_mazes"],
        noise_scale=exp_config["maze_noise_scale"],
        gaps=exp_config["maze_gaps"],
    )

    maze_test = multimodal_mazes.TrackMaze(
        size=exp_config["maze_size"], n_channels=len(exp_config["channels"])
    )
    maze_test.generate(
        number=1000,
        noise_scale=exp_config["maze_noise_scale"],
        gaps=exp_config["maze_gaps"],
    )

    # Agent
    if job_index != 1:
        n_hidden_units = 8
    else:
        n_hidden_units = 34

    agnt = multimodal_mazes.AgentDQN(
        location=None,
        channels=exp_config["channels"],
        sensor_noise_scale=exp_config["sensor_noise_scale"],
        n_hidden_units=n_hidden_units,
        wm_flags=wm_flag,
    )

    n_parameters = 0
    for p in agnt.parameters():
        n_parameters += p.numel()
    agnt.n_parameters = n_parameters

    # Train
    agnt.generate_policy(maze=maze, n_steps=exp_config["n_steps"], maze_test=maze_test)

    # Test
    results, input_sensitivity, memory = multimodal_mazes.test_dqn_agent(
        maze_test=maze_test,
        agnt=agnt,
        exp_config=exp_config,
        noises=noises,
    )

    # Store
    agnt.results = results
    agnt.input_sensitivity = input_sensitivity
    agnt.memory = memory

    return agnt


if __name__ == "__main__":
    # Config files
    exp_config = multimodal_mazes.load_exp_config("../exp_config.ini")

    # Create save folder
    save_folder = "../results/" + re.sub(r"\[.*?\]", "", os.environ["PBS_JOBID"])
    os.makedirs(save_folder, exist_ok=True)

    # Copy config files to save folder
    shutil.copyfile("../exp_config.ini", save_folder + "/exp_config.ini")

    # Run
    job_index = int(os.environ["PBS_ARRAY_INDEX"]) - 1  # array job
    agnt = run_exp(job_index=job_index, exp_config=exp_config)

    # Save results
    save_path = save_folder + "/" + str(job_index)
    with open(save_path + ".pickle", "wb") as file:
        pickle.dump(agnt, file)
