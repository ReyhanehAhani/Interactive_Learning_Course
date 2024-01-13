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
This section runs the SARSA algorithm, which is another reinforcement learning algorithm that updates Q-values based on the current policy.

Each of the training loops iterates through episodes and updates the Q-values based on the observed rewards and transitions in the environment. The results are stored in arrays for later analysis or plotting.

# Deep RL Hands-on
## Linear Function Approximator

## Introduction
This code implements a simple linear function approximator with examples demonstrating its usage in two scenarios: approximating a linear function and approximating the sine function. The code also includes a section on Policy Gradient in Reinforcement Learning and an implementation of the Deep SARSA algorithm.

## Linear Function Approximator
### Implementation
The `LinearFunctionApproximator` class is implemented to approximate a linear function. The class has methods for prediction (`predict`) and updating the model (`update`). The example usage section demonstrates the training process on two different linear functions.

### Example Usage
1. **Linear Function Approximation:**
   ```python
   num_features = 2
   approximator = LinearFunctionApproximator(num_features)
   num_episodes = 1000

   for episode in range(num_episodes):
       phi_s = np.random.rand(num_features)
       target_value = 10 * phi_s[0] + 5 * phi_s[1]
       approximator.update(phi_s, target_value)

   test = np.array([0.8, 0.2])
   predicted_value = approximator.predict(test)
   print(f"Predicted Value for phi_s {test}: {predicted_value}")
   ```

2. **Sine Function Approximation:**
   ```python
   num_terms = 3
   approximator = LinearFunctionApproximator(num_terms)
   num_episodes = 1000

   for episode in range(num_episodes):
       x = np.random.uniform(-np.pi, np.pi)
       features = np.array([x**i / np.math.factorial(i) for i in range(1, 2 * num_terms, 2)])
       target_value = np.sin(x)
       approximator.update(features, target_value)

   test_x_values = np.linspace(-np.pi, np.pi, 100)
   predicted_values = [approximator.predict([x**i / np.math.factorial(i) for i in range(1, 2 * num_terms, 2)]) for x in test_x_values]

   plt.plot(test_x_values, np.sin(test_x_values), label='True sin(x)')
   plt.plot(test_x_values, predicted_values, label='Approximation')
   plt.legend()
   plt.title('Linear Approximation of sin(x)')
   plt.xlabel('x')
   plt.ylabel('sin(x)')
   plt.show()
   ```

## Policy Gradient in Reinforcement Learning
### Implementation
This section focuses on the implementation of the Policy Gradient algorithm in reinforcement learning using PyTorch. The code defines a neural network (`Policy`) representing the policy and provides functions for training and evaluating the agent.

### Example Usage
1. **Training a CartPole Agent:**
   ```python
   # Define parameters
   cartpole_hyperparameters = {
       "h_size": 16,
       "n_training_episodes": 1000,
       "n_evaluation_episodes": 10,
       "max_t": 1000,
       "gamma": 1.0,
       "lr": 1e-2,
       "env_id": env_id,
       "state_space": s_size,
       "action_space": a_size,
   }

   # Create policy and optimizer
   cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
   cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

   # Train the agent
   scores = reinforce(cartpole_policy, cartpole_optimizer, cartpole_hyperparameters["n_training_episodes"], cartpole_hyperparameters["max_t"], cartpole_hyperparameters["gamma"], 100)

   # Plot the training progress
   plt.plot(scores)
   ```

2. **Evaluating the Trained Agent:**
   ```python
   # Evaluate the trained agent
   mean_reward, std_reward = evaluate_agent(eval_env, cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"], cartpole_policy)
   print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
   ```

3. **Recording a Video of the Trained Agent:**
   ```python
   # Record a video of the trained agent
   video_path = 'cartpole_video'
   record_video(env, cartpole_policy, video_path, 30)
   ```

## Deep SARSA
### Implementation
This section provides an implementation of the Deep SARSA algorithm using PyTorch. It includes a neural network (`SARSAPolicy`) to represent the policy and a training loop to update the model.

### Example Usage
```python
# Create SARSA policy and optimizer
sarsa_policy = SARSAPolicy(sarsa_hyperparameters["state_space"], sarsa_hyperparameters["action_space"], sarsa_hyperparameters["h_size"]).to(device)
sarsa_optimizer = optim.Adam(sarsa_policy.parameters(), lr=sarsa_hyperparameters["lr"])

# SARSA training loop
for episode in range(sarsa_hyperparameters["n_training_episodes"]):
    # ... (training loop details)
```

## Visualization
The code also includes visualization components, such as plotting training progress and recording videos of the trained agents.

