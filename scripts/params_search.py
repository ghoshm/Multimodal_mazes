# Params search

import numpy as np
import os
import shutil

import multimodal_mazes


if __name__ == "__main__":
    # Config files
    search_config = multimodal_mazes.load_search_config("../search_config.ini")

    # Create save folder
    os.makedirs(search_config["save_path"], exist_ok=True)

    # Copy config file to save folder
    shutil.copyfile(
        "../search_config.ini", search_config["save_path"] + "/search_config.ini"
    )

    # Run search
    results, parameters = multimodal_mazes.prey_params_search(
        grid_size=search_config["grid_size"],
        size=search_config["size"],
        n_prey=search_config["n_prey"],
        n_steps=search_config["n_steps"],
        n_trials=search_config["n_trials"],
        pk=search_config["pk"],
        scenario=search_config["scenario"],
        motion=search_config["motion"],
    )

    # Save results
    np.save(search_config["save_path"] + "/results", results)
    np.save(search_config["save_path"] + "/parameters", parameters)
