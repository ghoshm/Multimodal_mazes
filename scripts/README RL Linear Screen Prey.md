# Result Notebook Documentation

<p>This notebook implements Reinforcement Learning (RL) experiments using Q-learning for agent training in multi-modal mazes. The agent's goal is to capture or approach prey under various scenarios and motion conditions. The agent is evaluated on both capture and approach success across multiple trials and training parameters. Key features include RL training, feature importance heatmaps, and visualization of results based on different scenarios.</p>

<p>The results of these simulations, including performance metrics for different agents and policies, are stored in a structure called 'RL Results' for further analysis and retrieval.</p>

## Imports
The necessary packages for RL training, environment evaluation, visualization, and progress tracking are imported:

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multimodal_mazes  # Custom module for RL environment
import seaborn as sns
```

## Parameters and Helper Functions

### Actions
The `actions` dictionary defines eight possible movement actions for the agent in the environment.

### General Parameters
- `width`, `height`: Dimensions of the environment.
- `pk`: Patch size for local sensory information.
- `agent_location`: Initial location of the agent.
- `n_prey`: Number of prey in the environment.
- `n_steps`, `n_trials`: Number of steps per trial and trials per scenario.
- `n_features`: Number of features observed by the agent.
- `cost_per_step`, `cost_per_collision`: Penalty values for steps and collisions.
- `alpha`, `epsilon`, `gamma`: RL learning rate, exploration rate, and discount factor, respectively.

### Static and Constant Hyperparameters Functions
Two helper functions define the hyperparameters for the static and constant motion scenarios.

```python
def static_hyperparamters():
    # Defines hyperparameters for the static scenario.
```

```python
def constant_hyperparameters():
    # Defines hyperparameters for the constant motion scenario.
```

### Plot Feature Importance Heatmap
This function visualizes the importance of each feature in the agent's decision-making process.

```python
def plot_feature_importance_heatmap(agent):
    # Generates a heatmap displaying the relative importance of features used by the agent.
```

## RL Agent Training and Evaluation

### Static Training Test
- A Q-Learner agent (`QLearnerAgent`) is instantiated with the static scenario parameters.
- The agent is trained using `train_RL()` over 2000 trials.
- Evaluation plots for training performance are generated, followed by a feature importance heatmap.

```python
training_evaluator.train_RL(training_trials=2000)
```

### Constant Movement Training Test
- The agent is trained under a constant movement scenario, where prey follows linear motion.
- Training is conducted for 5000 trials with visualizations for capture success and agent performance.

```python
training_evaluator.train_RL(training_trials=5000)
```

## Success vs Speed Experiment
### Results Storage and Hyperparameters
This section conducts experiments to test the success of the agent in capturing prey at varying speeds (`pms`) under different cases.

### Agent Training and Testing
The agent is trained and tested across four different agents for various scenarios. Each agent evaluates the prey capture and approach success for different parameter sets.

### Results Processing and Plotting
The capture and approach success results are processed, averaged, and plotted with error bars to showcase performance trends.

## Curriculum Learning Success vs Speed
In this section, agents are trained using curriculum learning, with increasingly difficult tasks as training progresses. The results are plotted similarly to the Success vs Speed experiment, comparing different speeds and task cases.

## Disappearing Prey Experiment
This experiment evaluates agent performance when prey disappears after a certain number of steps (`times_to_disappear`). Multiple agents are trained to adapt to this challenge, and results are plotted based on capture and approach success.

---

### Key Functions:
- `train_RL()`: Trains the agent for a specified number of trials.
- `evaluate()`: Evaluates the agent's performance in capturing or approaching prey under varying conditions.
- `training_plots()`: Visualizes training performance, including capture success and feature importance.
- `plot_feature_importance_heatmap()`: Displays feature importance for the agent's decision-making process.

### Key Metrics:
- **Capture Success**: Percentage of trials where the agent successfully captures prey.
- **Approach Success**: Percentage of trials where the agent approaches the prey within a certain proximity.
- **Feature Importance**: Analysis of feature weights influencing the agent's action choices.

This notebook systematically trains and evaluates the RL agent under various challenging scenarios to measure its performance in dynamic environments.