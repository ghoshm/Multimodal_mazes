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
    }

    return exp_config_dict
