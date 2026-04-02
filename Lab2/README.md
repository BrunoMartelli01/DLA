
---

# Lab 2: Deep Reinforcement Learning Laboratory

## Overview
This laboratory is divided into three main exercises, progressing from basic **Policy Gradient** methods to advanced **Value-Based** methods using Convolutional Neural Networks (CNNs).

---

## Exercise 1: Improving REINFORCE (Warm-up)
**Objective:** Implement the basic REINFORCE algorithm to solve the `CartPole-v1` environment.

*   **Key Components:**
    *   **`PolicyNet`**: A simple feed-forward neural network that outputs action probabilities using a Softmax layer.
    *   **Stochastic Action Selection**: Sampling actions from a categorical distribution to encourage exploration.
    *   **Reward Computation**: Implementing discounted returns ($G_t$) to assign credit to past actions.
*   **Outcome:** Understanding how gradient ascent on the log-probability of actions, weighted by returns, allows an agent to learn a policy from scratch.

---

## Exercise 2: REINFORCE with a Value Baseline
**Objective:** Reduce the high variance of the REINFORCE algorithm by introducing a learned baseline.

*   **Key Components:**
    *   **`ValueNet`**: A secondary network that estimates the state-value function $V(s)$.
    *   **Advantage Calculation**: Updating the policy using the **Advantage** ($G_t - V(s)$) instead of raw returns. This tells the agent if an outcome was better or worse than *expected*.
    *   **Monte Carlo Comparison**: A statistical experiment (10 independent runs) comparing three strategies:
        1.  **Standard REINFORCE**: No normalization.
        2.  **Normalized REINFORCE**: Standardizing returns within an episode.
        3.  **Baseline (Actor-Critic)**: Using the Value Network to stabilize updates.
*   **Outcome:** Demonstrating that the **Baseline method** provides the fastest and most stable convergence.

---

## Exercise 3: Deep Q-Learning (DQL) for CarRacing
**Objective:** Solve the vision-based `CarRacing-v3` environment using a Deep Q-Network (DQN).

*   **Logic Location:** Classes and architecture in `DQL.py`; 
*   **Key Components:**
    *   **`QNetwork` (CNN)**: A 3-layer Convolutional Neural Network that processes 4 stacked grayscale frames to detect spatial features and motion.
    *   **DQN Stability**: Implementation of a **Replay Buffer** (to break temporal correlation) and a **Target Network** .
    *   **Environment Specialization (Wrappers)**:
        *   `CropObservation`: Removes the dashboard to focus strictly on the track.
        *   `StayOnTrack`: A custom reward-shaping wrapper that penalizes driving on grass and rewards staying on the asphalt.
*   **Outcome:** A trained agent capable of navigating a complex racing track using discrete actions.

---

## File Structure & Usage

### 1. `DQL.py` 
This file acts as a module. It contains all the necessary classes for Exercise 3:
*   `QNetwork`, `DQNAgent`, `ReplayBuffer`.
*   Preprocessing wrappers: `StayOnTrack`, `CropObservation`.

### 2. `Lab2.ipynb` 
This is the main entry point. Running this notebook to executes the exercises in sequence:
```bash
jupyter notebook  .\Lab2\Lab2.ipynb
```
*   It performs the **REINFORCE** training (Ex 1 & 2).
*   It generates the **Monte Carlo** comparison plots.
*   It imports the `DQNAgent` from `DQL.py` to start the **CarRacing** training (Ex 3).

### 3. Monitoring & Results
*   **TensorBoard**: Track DQL training progress (Reward, Loss, Epsilon) for Exercise 3.
    ```bash
    tensorboard --logdir=.\Lab2\runs
    ```
*   **Checkpoints**: Best performing models for Exercise 3 are saved in the `./dqn/` directory.

---


## Results Summary
*   **REINFORCE**: Shows that while simple to implement, it is highly sensitive to noise. The **Baseline** variant is the only one that consistently and rapidly solves the `CartPole-v1` environment.
*   **DQL**: The combination of **Frame Stacking** and **Reward Shaping** (via `StayOnTrack`) is essential for the agent to learn the concept of a "track" rather than spinning in circles or driving off-road.

---