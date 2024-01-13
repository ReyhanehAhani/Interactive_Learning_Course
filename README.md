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

Feel free to explore and modify the code for your specific use case. If you have any questions or suggestions, please create an issue or reach out to the repository owner.

**Happy experimenting!**
