# 🚀 Comparing PPO and DQN on LunarLander-v3 🌕

This project trains and evaluates two powerful reinforcement learning agents, using **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)**, to master the `LunarLander-v3` environment from Gymnasium. The objective is to land a lunar module safely and efficiently between two flags on the moon's surface.

## Table of Contents
- [Techniques Used](#-techniques-used)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Results](#-results)
  - [Learning Curve](#learning-curve)
  - [Agent Performance](#agent-performance)
- [File Structure](#-file-structure)


## 🛠️ Techniques Used

This project leverages several key concepts and libraries in modern reinforcement learning:

*   **Algorithms:**
    *   ✅ **Proximal Policy Optimization (PPO):** An on-policy, actor-critic algorithm renowned for its stability and excellent performance. It learns a policy and a value function, using a clipped objective to prevent overly large policy updates that could destabilize training.
    *   ✅ **Deep Q-Network (DQN):** An off-policy, value-based algorithm that learns the optimal action-value function (Q-function). This implementation uses a replay buffer to break the correlation between consecutive experiences and a target network for stable learning.

*   **Frameworks:**
    *   ✅ **Stable Baselines3:** A comprehensive library of reliable, high-quality implementations of reinforcement learning algorithms in PyTorch.
    *   ✅ **Gymnasium:** The industry-standard toolkit for developing and comparing RL algorithms, providing the `LunarLander-v3` environment.

*   **Best Practices:**
    *   ✅ **`EvalCallback`:** To ensure we save the best possible agent, a callback is used during training to periodically evaluate its performance on a separate, deterministic environment. This captures the agent at its peak skill level and helps avoid issues with overfitting or catastrophic forgetting.
    *   ✅ **Vectorized Environments:** The training process is accelerated by using `make_vec_env` to run 16 environments in parallel, allowing for much faster data collection and training iterations.


## ⚙️ Hyperparameter Tuning

Hyperparameter tuning is one of the most critical steps for achieving high performance in reinforcement learning. It involves adjusting the learning parameters of the algorithm to find a balance between exploration (trying new actions) and exploitation (using known good actions).

The values below were selected as a strong starting point for this environment, leading to the successful results shown.

### PPO Hyperparameters

| Hyperparameter   | Value     | Description                                             |
| ---------------- | --------- | ------------------------------------------------------- |
| `policy`         | `MlpPolicy` | Standard Multi-Layer Perceptron (neural network) policy.|
| `n_steps`        | `1024`    | Steps to collect from each environment before an update.|
| `batch_size`     | `64`      | Minibatch size for each gradient update.                |
| `n_epochs`       | `4`       | Number of optimization epochs per update.               |
| `gamma`          | `0.999`   | Discount factor for future rewards.                     |
| `gae_lambda`     | `0.98`    | Factor for Generalized Advantage Estimation (GAE).      |
| `ent_coef`       | `0.01`    | Entropy coefficient to encourage exploration.           |
| `learning_rate`  | `3e-4`    | The step size for the optimizer (Adam).                 |

### DQN Hyperparameters

| Hyperparameter          | Value       | Description                                                 |
| ----------------------- | ----------- | ----------------------------------------------------------- |
| `policy`                | `MlpPolicy` | Standard Multi-Layer Perceptron (neural network) policy.    |
| `learning_rate`         | `6.3e-4`    | The step size for the optimizer (Adam).                     |
| `buffer_size`           | `50000`     | The size of the experience replay buffer.                   |
| `learning_starts`       | `1000`      | Number of steps to collect before training starts.          |
| `batch_size`            | `128`       | Minibatch size sampled from the replay buffer.              |
| `gamma`                 | `0.99`      | Discount factor for future rewards.                         |
| `train_freq`            | `4`         | Update the model every 4 steps.                             |
| `exploration_fraction`  | `0.12`      | Fraction of training time to spend decreasing epsilon.      |
| `exploration_final_eps` | `0.1`       | The final value of epsilon (random action probability).     |


## 📊 Results

### Learning Curve

The plot below illustrates the learning progress of the PPO and DQN agents over 2,500,000 training timesteps. The reward is smoothed to better visualize the underlying performance trend.

PPO demonstrates faster initial learning and converges to a slightly higher and more stable final reward compared to DQN in this experiment.

![Learning Curve](results/plots/learning_curve.png)

### Agent Performance

The final saved models represent the agents at their peak performance, as captured by the `EvalCallback`.

#### PPO Agent in Action
The PPO agent learns a very smooth and fuel-efficient landing strategy.

![PPO Agent Demo](results/videos/ppo_demo.gif)

#### DQN Agent in Action
The DQN agent also learns a successful landing policy, occasionally exhibiting a more "decisive" or twitchy control style as it follows its learned Q-values.

![DQN Agent Demo](results/videos/dqn_demo.gif)


## 📁 File Structure

```
.
├── 📄 .gitignore
├── 📄 README.md
├── 📄 requirements.txt
├── 🐍 lunar_lander_rl.py
└── 📂 results/
    ├── 📂 plots/
    │   └── 🖼️ learning_curve.png
    └── 📂 videos/
        ├── 🎬 ppo_demo.gif
        └── 🎬 dqn_demo.gif
```
