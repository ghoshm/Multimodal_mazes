# Multimodal mazes

A library for creating multisensory environments and testing various agents. 

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
pip install -e . 
```

## Usage

Broadly, the library consists of **tasks** and **agents**. 

Note that the notebooks in scripts (*.ipynb*) are intended for analysis and working, and may not run from start to finish.   

### Tasks 

There are two types of tasks:
* **Predator-prey** scenarios - in which agents must hunt prey who move and emit sensory cues (e.g. visual and auditory signals).  

* **Mazes** - in which agents must use sensory cues from multiple channels (e.g. vision and hearing) to reach a goal location.

To create a batch of mazes: 
```python
import multimodal_mazes

maze = multimodal_mazes.GeneralMaze(size=33, n_channels=2)
maze.generate(number=10, noise_scale=0.0)
``` 

Here, the maze object stores the mazes themselves (*maze.mazes*) and other useful things. For example, each maze's start and goal locations (*maze.start_locations* and *maze.goal_locations*), and the time it would take to traverse each maze's shortest path (*maze.fastest_solutions*).

Each maze itself (e.g. *maze.mazes[0]*) is a Numpy array of shape size x size x (n_channels + 1) where: 
* *maze.mazes[0][:,:,-1]* stores the maze's structure: walls (0.0), track (1.0).
* *maze.mazes[0][:,:,:-1]* stores each channel's sensory cues. For example, gradients leading towards the goal. 

### Agents 

Agents navigate environments. To do so, they:

1. Sense their local environment.
2. Implement a **policy** mapping their sensations to actions.
3. Act. By default, actions include moving in a cardinal direction, or pausing. Note that when an agent tries to move into a wall, it will simply remain in place. 

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
time, path = multimodal_mazes.maze_trial(mz=maze.mazes[0], mz_start_loc=maze.start_locations[0], mz_goal_loc=maze.goal_locations[0], channels=[1,1], sensor_noise_scale=0.0, drop_connect_p=0.0, n_steps=100, agnt=agnt) 

multimodal_mazes.plot_path(path, mz=maze.mazes[0], mz_goal_loc=maze.goal_locations[0], n_steps=100)
```

To test an agent on a batch of mazes: 
```python
fitness = multimodal_mazes.eval_fitness(genome=None, config=None, channels=[1,1], sensor_noise_scale=0.0, drop_connect_p=0.0, maze=maze, n_steps=100, agnt=agnt)
```

Each agent's fitness is scored from 0 to 1 (with higher indicating better performance). On maze tasks, a value of 1 means the agent always takes the shortest path.

## Training 

To train a neural network to perform a task, the library supports using either:

* NEAT - to co-learn neural network architectures and weights. 

* DQN - to learn the weights for a given network architecture. 

In each of these cases: open a terminal, change into the *scripts* folder and then run either: *maze_exp.py* (NEAT) or *DQN_exp.py* (DQN).

Both:
* Read configuration files: NEAT - *neat_config.ini* and *exp_config.ini*. DQN - *../exp_config.ini*. 

* Create a save folder to store copies of these configuration files and their output results.  












