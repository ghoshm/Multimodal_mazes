# Result Notebook Documentation
<p>This script simulates predator-prey interactions in a multimodal maze environment. The predator's task is to capture prey, with various configurations of prey motion, multisensory inputs, and agent policies. The agent's behavior is modeled using rule-based, memory-based, or random policies. The script also includes visualizations of the predator-prey interactions and performance metrics, such as percentage capture and approach.</p>

<p>The results of these simulations, including performance metrics for different agents and policies, are stored in a structure called 'Rulebased Results' for further analysis and retrieval.</p>
---

#### Key Components

1. **Imports**:
   - Standard libraries like `numpy`, `pickle`, and `matplotlib` are used for numerical operations, data handling, and visualizations.
   - `multimodal_mazes` is the core module providing the agent behaviors, environment settings, and prey simulation functions.
   - `tqdm` is used for progress tracking in loops.

2. **Hyperparameters**:
   - **width/height**: Dimensions of the maze environment.
   - **n_prey**: Number of prey in the environment.
   - **n_steps**: Number of steps in each simulation trial.
   - **n_trials**: Number of trials to evaluate the agent's performance.
   - **pk**: Width of the prey’s Gaussian signal.
   - **scenario, motion, multisensory**: Define the type of simulation scenario (e.g., Constant, Random, Two Prey), prey motion type, and whether the agent receives multisensory or unisensory inputs.

3. **Agent and Prey Initialization**:
   - Agents can have different policies (rule-based, memory-based, or random) with varying strategies for capturing prey.
   - Prey behavior can be static or dynamic, with different noise levels and motion types.

4. **Simulation Execution**:
   - The core simulation is performed by `PredatorTrial` or similar classes in the `multimodal_mazes` package. The trial runs and logs the agent's path, prey state, and environment states.

5. **Visualization**:
   - The script includes extensive plotting functionalities:
     - Animations of the agent’s pursuit of prey.
     - Visualization of sensory inputs using colormaps.
     - Plots showing performance metrics such as capture percentage vs. speed and approach percentage vs. prey visibility.

6. **Performance Metrics**:
   - **Percentage Capture vs. Speed**: Tests how well different policies capture prey at varying speeds.
   - **Percentage Approached vs. Time**: Shows how prey visibility affects the agent’s approach percentage.
   - **Two Prey Capture Probability**: Evaluates the probability of capturing multisensory prey first under different noise conditions.

---

#### Functions and Visualization Techniques

- **update_animation(t)**: Updates the environment and agent-prey positions for each frame in the animation.
  
- **Evaluator Class**: Provides fitness evaluation for the agent's behavior, calculating metrics like capture and approach percentage for multiple trials.

- **Colormap Integration**: Uses custom colormaps to represent different sensory channels and environmental factors, providing a clear visual representation of agent behavior.

- **Performance Visualization**:
   - **Line plots** for percentage capture vs. speed, approach percentage vs. time, and multisensory prey capture probability.
   - **Error bars** represent the variability across multiple simulation runs (standard deviation).
  
---

#### Key Parameters for Customization

- **Scenario and Motion Types**:
   - Scenario options include static, constant, random, and two-prey situations.
   - Motion can be linear, Levy, or custom types for prey movements.
  
- **Agent Policies**:
   - Policies include several predefined rule-based, memory-based, and random strategies, such as "Nonlinear fusion," "Levy motion," or "Kinetic alignment."

- **Multisensory or Unisensory Inputs**:
   - The agent can be tested in different sensory environments (e.g., unisensory, balanced multisensory) to explore how different sensory strategies affect predator-prey dynamics.

---

#### Conclusion

This script simulates predator-prey interactions in various complex environments and provides both qualitative (visualization) and quantitative (metrics) insights into agent performance. It is highly customizable, with options for adjusting scenario settings, agent policies, and sensory conditions to explore a wide range of behavioral dynamics in the simulated environment.