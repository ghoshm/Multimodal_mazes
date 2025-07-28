# Multimodal mazes

A library for creating multisensory environments and testing various agents. 

![Multimodal mazes](https://github.com/ghoshm/Multimodal_mazes/readme_images/MM_logo.png)

## Setup 

To install: 
1. Clone this repo. 

2. Create a virtual environment with the necessary dependencies (listed in *environment.yml*). To do this in [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), open the Multimodal_mazes folder in a terminal and run:
```
conda env create -f environment.yml
```
Note that you may need to pip install some packages (e.g. [neat-python](https://neat-python.readthedocs.io/en/latest/)). 

3. Finally, install the multimodal mazes package itself, by running: 
```
conda activate Multimodal_mazes
pip install -e . 
```

## Usage

Broadly, the library consists of **tasks** and **agents**. 

Note that the notebooks (*.ipynb*) in scripts are intended for analysis, plotting and working, and not for running from start to finish. 

### Tasks 

There are two types of tasks:
* **Predator-prey** scenarios - in which agents must hunt prey who move and emit sensory cues (e.g. visual and auditory signals).  

* **Mazes** - in which agents must use sensory cues from multiple channels (e.g. vision and hearing) to reach a goal location.

To create a batch of mazes: 
```python
import multimodal_mazes

maze = multimodal_mazes.GeneralMaze(size=33, n_channels=2)
maze.generate(number=10)
``` 

Here, the maze object stores the mazes themselves (*maze.mazes*) and other useful things: each maze's start and goal locations (*maze.start_locations* and *maze.goal_locations*), and the time it would take to traverse each maze's shortest path (*maze.fastest_solutions*).

Each maze itself (*maze.mazes[n]*) is a Numpy array of shape size x size x (n_channels + 1) where: 
* *maze.mazes[n][:,:,-1]* stores the maze's structure: walls (0.0) and paths (1.0).
* *maze.mazes[n][:,:,:-1]* stores each channel's sensory cues. For example, gradients leading towards the goal. 

![Maze structure](https://github.com/ghoshm/Multimodal_mazes/readme_images/Mz_structure.png)

### Agents 

Agents navigate environments. To do so, they:

1. Sense their local environment.
2. Implement a **policy** mapping their sensations to actions.
3. Act. By default, actions include moving in a cardinal direction, or pausing. Note that when an agent tries to move into a wall, it will simply remain in place. 

![Agent sensation-action loop](https://github.com/ghoshm/Multimodal_mazes/readme_images/Sense_policy_act.png)

All agents share the same sensations and actions (inherited from the *Agent* class) but use different policies. These include: 
* Rule-based policies. For example, sum your sensory inputs across channels and move in the direction with the greatest total (linear fusion).   
* Neural networks evolved using the NEAT algorithm. 
* Neural networks trained using reinforcement learning. 

To create an agent: 
```python
agnt = multimodal_mazes.AgentRuleBased(location=None, channels=[1,1], policy="Linear fusion")
```

## Testing

To test an agent on a single maze, and plot it's path:
```python
n = 0 # select a maze
time, path = multimodal_mazes.maze_trial(mz=maze.mazes[n], mz_start_loc=maze.start_locations[n], mz_goal_loc=maze.goal_locations[n], channels=[1,1], sensor_noise_scale=0.0, drop_connect_p=0.0, n_steps=100, agnt=agnt) 

multimodal_mazes.plot_path(path, mz=maze.mazes[n], mz_goal_loc=maze.goal_locations[n], n_steps=100, style="gradients")
```

![Example agent path](https://github.com/ghoshm/Multimodal_mazes/readme_images/Mz_agnt_path.png)

To test an agent on a batch of mazes: 
```python
fitness = multimodal_mazes.eval_fitness(genome=None, config=None, channels=[1,1], sensor_noise_scale=0.0, drop_connect_p=0.0, maze=maze, n_steps=100, agnt=agnt)
print(fitness)
```

Each agent's fitness is scored from 0 to 1 (with higher indicating better performance). On maze tasks, a value of 1 means the agent always takes the shortest path.

## Training 

To train a neural network to perform a task, the library supports using either:

* NEAT - to co-learn neural network architectures and weights. 

* DQN - to learn the weights for a given network architecture. 

### Minimal example 
To train a DQN network locally: 
```python
import numpy as np
import multimodal_mazes

maze = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze.generate(number=1000)
agnt = multimodal_mazes.AgentDQN(location=None, channels=[1,1], sensor_noise_scale=0.05, n_hidden_units=8, wm_flags=np.array([0,0,0,0,0,0,0]))
agnt.generate_policy(maze=maze, n_steps=20, maze_test=None)
```

And to test it on another set of mazes: 
```python
maze_test = multimodal_mazes.GeneralMaze(size=9, n_channels=2)
maze_test.generate(number=1000)
fitness = multimodal_mazes.eval_fitness(genome=None, config=None, channels=[1,1], sensor_noise_scale=0.05,drop_connect_p=0.0, maze=maze_test, n_steps=20, agnt=agnt)
print(fitness)
```

Note that changing the (binary) values in the *wm_flags* variable will allow you to train different architectures.

### From the terminal
Open a terminal, change into the *scripts* folder and then run either *maze_exp.py* for NEAT or *DQN_exp.py* for DQN:

```
 python maze_exp.py 
 python DQN_exp.py 
```

Both:
* Read configuration files: NEAT - *neat_config.ini* and *exp_config.ini*. DQN - *../exp_config.ini*. 

* Create a save folder to store copies of these configuration files and their output results.  












