# Gradient Bandit Algorithm (HW2)

This repository contains an implementation of the Gradient Bandit Algorithm in Python. The algorithm is applied to a simulated environment where a student takes different actions, and the goal is to maximize cumulative rewards.

## Contents

### `main.py`

This script contains the core implementation. Let's break down the key components:

#### 1. `Student` Class

The `Student` class represents a student and defines the method `get_reward(action)` to simulate the reward received when the student takes a specific action. Rewards are generated using a truncated normal distribution with different parameters for each action.

#### 2. `Environment` Class

The `Environment` class represents the environment in which the student interacts. It takes a `Student` object as an argument and defines the method `calc_reward(action)` to calculate the reward for a given action by calling the corresponding method in the associated `Student` object.

#### 3. `Gradient_Bandit_agent` Class

The `Gradient_Bandit_agent` class implements the Gradient Bandit Algorithm. It takes an `Environment` object, learning rate, and the number of arms (actions) as parameters. The key method, `gradient_bandit_alg(num_iter)`, executes the algorithm for a specified number of iterations and returns softmax probabilities, reward history, and regret history.

#### 4. `plot` Function

The `plot` function visualizes the performance of the Gradient Bandit Algorithm for different learning rates. It generates plots showing cumulative rewards and regrets for 10 simulated schools, each with 52 weeks and 100 students. The mean and 95% confidence intervals of rewards and regrets across the schools are also calculated and plotted.

### Usage

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:

   ```bash
   cd gradient-bandit-algorithm
   ```

3. Run the `main.py` script:

   ```bash
   python main.py
   ```

4. Observe the generated plots, which illustrate the algorithm's performance for different learning rates.

### Examples

```python
# For learning_rate = 0.001
plot(env, 0.001)

# For learning_rate = 0.01
plot(env, 0.01)

# For learning_rate = 0.1
plot(env, 0.1)
```

### Results

The plots provide insights into how the Gradient Bandit Algorithm performs with different learning rates. The mean cumulative rewards and regrets, along with 95% confidence intervals, are displayed for 10 runs.

Sure, let's break down the provided code into sections and explain each part:


# Q-learning and SARSA (HW3)

### Libraries
```python
!pip install gymnasium
```
This installs the "gymnasium" library using the pip installer.

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
```
These lines import necessary libraries for numerical operations, the gymnasium reinforcement learning library, and specific modules from gymnasium.

### GYM Environment - DiscretizedObservationWrapper
```python
class DiscretizedObservationWrapper(gym.ObservationWrapper):
    # ... (constructor and other methods)
```
This class is a custom observation wrapper for the Gym environment. It discretizes continuous observations into discrete ones.

- `_convert_to_one_number`: A method to convert the discretized observations into a single number.
- `observation`: The method that gets called when the observation is requested. It transforms the continuous observations into discrete ones.
- `get_transition_matrix`: Generates transition dynamics similar to 'P' in standard Gym environments.
- `get_possible_next_states`: Determines possible next states based on the current state and action.

### GridWorldEnv - Custom Gym Environment
```python
class GridWorldEnv(gym.Env):
    # ... (constructor and other methods)
```
This class defines a custom Gym environment named GridWorldEnv. It represents a simple grid world where an agent moves towards a target while avoiding obstacles.

- `_get_obs`: Returns the observation (state) of the environment.
- `_idx_to_cord`: Converts an index to coordinates in the grid.
- `reset`: Initializes the environment at the beginning of an episode.
- `reward`: Computes the reward based on the agent's action and environment state.
- `step`: Performs one step of the environment dynamics based on the agent's action.

### Hyperparameters
```python
alpha = 0.1
gamma = 0.9
min_eps = 0.01
max_eps = 1.0
episodes = 750
iters = 10
seeds = np.random.randint(int(1e10), size=(iters, episodes))
epsilons = np.linspace(max_eps, min_eps, episodes) ** 3
```
These are hyperparameters for the reinforcement learning algorithms, such as learning rates (`alpha`), discount factor (`gamma`), minimum and maximum exploration rates (`min_eps` and `max_eps`), the total number of episodes, and random seeds for reproducibility.

### Plotting Epsilons over Episodes
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(epsilons, label="Epsilons")
plt.title('Epsilons over episodes')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.grid()
plt.legend()
ax = plt.gca()
ax.set_ylim([0, 1])
ax.set_xlim([0, episodes])
```
This section uses Matplotlib to plot the exploration rate (`epsilon`) over episodes.

### Q-Learning with Fixed Learning Rate
```python
max_fixed_rewards  = np.array([-np.inf] * episodes)
mean_fiexd_rewards = np.zeros(episodes)
# ... (Q-learning training loop)
```
This part of the code runs Q-learning with a fixed learning rate. It uses a Q-table to approximate the optimal action-value function.

### Q-Learning with Moving Learning Rate
```python
min_alpha = 0.1
max_alpha = 0.9
alphas = np.linspace(max_alpha, min_alpha, episodes) ** 2
max_moving_rewards = np.array([-np.inf] * episodes)
mean_moving_rewards = np.zeros(episodes)
# ... (Q-learning training loop with a moving learning rate)
```
Similar to the previous section, but here, Q-learning is implemented with a moving learning rate (`alpha`).

### Plotting Q-Learning Results
```python
plt.figure(figsize=(8, 5))
plt.plot(mean_fiexd_rewards, label="Fixed Learning Rewards (0.1)")
plt.title('Fixed results')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.grid()
plt.legend()
ax = plt.gca()
ax.set_ylim([-100, 3000])
ax.set_xlim([0, episodes])

plt.figure(figsize=(8, 5))
plt.plot(mean_moving_rewards, label="Moving Learning Rewards")
plt.title('Moving results')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.grid()
plt.legend()
ax = plt.gca()
ax.set_ylim([-100, 3000])
ax.set_xlim([0, episodes])
```
These plots display the rewards obtained during the Q-learning training with fixed and moving learning rates.

### SARSA
```python
mean_sarsa_rewards = np.zeros(episodes)
max_sarsa_rewards = np.array([-np.inf] * episodes)
# ... (SARSA training loop)
```
This

 section runs the SARSA algorithm, which is another reinforcement learning algorithm that updates Q-values based on the current policy.

Each of the training loops iterates through episodes and updates the Q-values based on the observed rewards and transitions in the environment. The results are stored in arrays for later analysis or plotting.
