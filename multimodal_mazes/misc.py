# Misc

import configparser


def load_exp_config(path):
    """
    Reads the experiment config file into a dictionary.
    Arguments:
        A path to an exp_config.ini file.
    Returns:
        A dictionary of hyperparameters.
    """
    exp_config = configparser.ConfigParser()
    exp_config.read(path)

    channels = list(map(int, (exp_config["DEFAULT"]["channels"]).split(",")))
    maze_size = int(exp_config["DEFAULT"]["maze_size"])
    n_mazes = int(exp_config["DEFAULT"]["n_mazes"])
    n_generations = int(exp_config["DEFAULT"]["n_generations"])
    n_steps = int(exp_config["DEFAULT"]["n_steps"])
    maze_noise_scale = float(exp_config["DEFAULT"]["maze_noise_scale"])
    maze_gaps = int(exp_config["DEFAULT"]["maze_gaps"])
    sensor_noise_scale = float(exp_config["DEFAULT"]["sensor_noise_scale"])
    drop_connect_p = float(exp_config["DEFAULT"]["drop_connect_p"])
    save_path = exp_config["DEFAULT"]["save_path"]
    maze_type = exp_config["DEFAULT"]["maze_type"]
    cue_sparsity = float(exp_config["DEFAULT"]["cue_sparsity"])
    wall_sparsity = float(exp_config["DEFAULT"]["wall_sparsity"])

    exp_config_dict = {
        "channels": channels,
        "maze_size": maze_size,
        "n_mazes": n_mazes,
        "n_generations": n_generations,
        "n_steps": n_steps,
        "maze_noise_scale": maze_noise_scale,
        "maze_gaps": maze_gaps,
        "sensor_noise_scale": sensor_noise_scale,
        "drop_connect_p": drop_connect_p,
        "save_path": save_path,
        "maze_type": maze_type,
        "cue_sparsity": cue_sparsity,
        "wall_sparsity": wall_sparsity,
    }

    return exp_config_dict


def load_search_config(path):
    """
    Reads the search config file into a dictionary.
    Arguments:
        A path to a search_config.ini file.
    Returns:
        A dictionary of hyperparameters.
    """
    search_config = configparser.ConfigParser()
    search_config.read(path)

    save_path = search_config["DEFAULT"]["save_path"]
    grid_size = int(search_config["DEFAULT"]["grid_size"])
    size = int(search_config["DEFAULT"]["size"])
    n_prey = int(search_config["DEFAULT"]["n_prey"])
    n_steps = int(search_config["DEFAULT"]["n_steps"])
    n_trials = int(search_config["DEFAULT"]["n_trials"])
    pk = int(search_config["DEFAULT"]["pk"])
    motion = search_config["DEFAULT"]["motion"]
    scenario = search_config["DEFAULT"]["scenario"]

    search_config_dict = {
        "save_path": save_path,
        "grid_size": grid_size,
        "size": size,
        "n_prey": n_prey,
        "n_steps": n_steps,
        "n_trials": n_trials,
        "pk": pk,
        "motion": motion,
        "scenario": scenario,
    }

    return search_config_dict


def load_prey_config(path):
    """
    Reads the prey config file into a dictionary.
    Arguments:
        A path to a prey_config.ini file.
    Returns:
        A dictionary of hyperparameters.
    """
    prey_config = configparser.ConfigParser()
    prey_config.read(path)

    save_path = prey_config["DEFAULT"]["save_path"]
    n_generations = int(prey_config["DEFAULT"]["n_generations"])
    n_trials = int(prey_config["DEFAULT"]["n_trials"])
    size = int(prey_config["DEFAULT"]["size"])
    sensor_noise_scale = float(prey_config["DEFAULT"]["sensor_noise_scale"])
    n_prey = int(prey_config["DEFAULT"]["n_prey"])
    pk = int(prey_config["DEFAULT"]["pk"])
    n_steps = int(prey_config["DEFAULT"]["n_steps"])
    scenario = prey_config["DEFAULT"]["scenario"]
    motion = prey_config["DEFAULT"]["motion"]
    pc = float(prey_config["DEFAULT"]["pc"])
    pm = float(prey_config["DEFAULT"]["pm"])
    pe = float(prey_config["DEFAULT"]["pe"])
    channels = list(map(int, (prey_config["DEFAULT"]["channels"]).split(",")))
    drop_connect_p = float(prey_config["DEFAULT"]["drop_connect_p"])

    prey_config_dict = {
        "save_path": save_path,
        "n_generations": n_generations,
        "n_trials": n_trials,
        "size": size,
        "sensor_noise_scale": sensor_noise_scale,
        "n_prey": n_prey,
        "pk": pk,
        "n_steps": n_steps,
        "scenario": scenario,
        "motion": motion,
        "pc": pc,
        "pm": pm,
        "pe": pe,
        "channels": channels,
        "drop_connect_p": drop_connect_p,
    }

    return prey_config_dict
