# Lab 2 Reinforcement Learning: DQL

## Overview

This laboratory explores two state-of-the-art reinforcement learning algorithms: Deep Q-Learning (DQL). 

---

## The Lab2.ipynb Notebook: Your Interactive Learning Platform

The **Lab2.ipynb** notebook is the core of this laboratory experience. It provides a comprehensive, step-by-step guided exploration of reinforcement learning algorithms with extensive explanations and visualizations.

### What You'll Find in the Notebook

The notebook is structured into detailed sections that take you through:

#### 1. **REINFORCE Algorithm - Warm-up Exercise**
- Hands-on implementation of the REINFORCE algorithm
- Interactive environment exploration with CartPole-v1
- Understanding the policy gradient method from first principles
- Step-by-step breakdown of key components:
  - **PolicyNet**: Neural network architecture for learning optimal policies
  - **Action Selection**: Stochastic policy implementation with probability distributions
  - **Returns Computation**: Calculating discounted cumulative rewards
  - **Episode Execution**: Complete training loop implementation
  - **Model Evaluation**: Robust performance assessment strategies

#### 2. **REINFORCE Implementation Details**
Each component is thoroughly explained with:
- **Purpose**: What the function achieves
- **Method**: How it implements the algorithm
- **Result**: Expected outputs and behaviors

Key sections include:
- Building neural networks that learn action-value functions
- Implementing epsilon-greedy exploration strategies
- Understanding the credit assignment problem
- Advantage normalization techniques
- Early stopping mechanisms for efficient training

#### 3. **Visualization and Analysis**
The notebook provides real-time plotting capabilities showing:
- Episode reward progression over time
- Training stability metrics
- Convergence speed analysis

#### 4. **Interactive Experimentation**
Unlike the standalone scripts, the notebook allows you to:
- Modify hyperparameters and immediately see results
- Run partial code cells for debugging and understanding
- Create custom visualizations
- Test different network architectures
- Experiment with various environments (CartPole-v1, LunarLander-v2, MountainCar-v0)

---

## Exercise 1: Deep Q-Learning (DQL)

### Objective
Implement and evaluate the Deep Q-Learning algorithm on classic control problems with discrete actions.

### Method
- Value-based RL agent using Q-learning with experience replay and target networks
- Epsilon-greedy exploration with decay
- Training and evaluation on environments like CartPole-v1, LunarLander-v2, and MountainCar-v0

### Using the Files

**Training the DQL agent:**
```bash
python DQL.py
```
This script:
- Initializes the environment and Q-network neural network
- Trains the agent using experience replay
- Saves model checkpoints during training
- Generates logs for TensorBoard monitoring


### Key Parameters
```python
learning_rate = 1e-3      # Learning rate
buffer_size = 100000      # Replay buffer size
batch_size = 64           # Training batch size
gamma = 0.99              # Discount factor
epsilon_decay = 0.995     # Exploration decay rate
```

### Expected Results
- DQL learns efficient policies on simpler discrete environments
- High sample efficiency but greater sensitivity to hyperparameters
- Training curves may show higher variance

---

**Installing dependencies:**
```bash
pip install -r ../requirements.txt
```

The requirements.txt file includes all necessary libraries such as:
- gymnasium (RL environments)
- torch (deep learning)
- tensorboard (visualization)

---

## File Structure

- `DQL.py`: Deep Q-Learning algorithm implementation
- `Lab2.ipynb`: **Interactive Jupyter notebook with all exercises and detailed explanations**
- `../runs/`: Directory containing training logs for TensorBoard
- `../dqn/`: where DQL model checkpoints are saved

---
