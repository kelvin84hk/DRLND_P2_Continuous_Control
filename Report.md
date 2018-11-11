# Introduction

The goal of this project is to train an agent deep reinforcement learning algorithms so that it can control the robot arm to touch the ball which is moving around it as long as it can.

# Implemention

Distributed Distributional Deterministic Policy Gradients (D4PG) have been implemented.

## Distributed Distributional Deterministic Policy Gradients (D4PG)

![alt text](https://github.com/kelvin84hk/DRLND_P2_Continuous_Control/blob/master/pics/d4pg_algo.jpg)

## Hyperparamters

After exploring several combinations, values below for hyperparameters allows the agent to solve the problem in stable manner.

Hyperparameter | Value
--- | ---    
Batch size | 64
Gamma | 0.99
τ | 1e-3
LR_ACTOR | 1e-4
LR_CRITIC | 1e-4
WEIGHT_DECAY | 0
N_step | 10
UPDATE_EVERY | 5
Vmax | 5
Vmin | 0
N_ATOMS | 51

## Network Structures

### Actor

Layer | Dimension
--- | ---
Input | N x 33
Linear Layer, Leaky Relu | N x 256
Linear Layer, Leaky Relu | N x 4
Batchnormalization1D | N x 4
Tanh Output | N x 4

### Critic

Layer | Dimension
--- | ---
Input | N x 33
Linear Layer, Leaky Relu | N x 128
Linear Layer + Actor Output, Leaky Relu | N x (128 + 4)
Linear Layer, Leaky Relu | N x 128
Linear Layer | N x 51

## Training Results

Below are the number of episodes needed to solve the environment and the evolution of rewards per episode during training.

![alt text](https://github.com/kelvin84hk/DRLND_P2_Continuous_Control/blob/master/pics/d4pg_results.png)

Environment was solved in 3327 episodes with Average Score 30.05.

# Ideas for future work

1. Implementing Mixture of Gaussians for action-value distribution described in original D4PG paper [https://arxiv.org/pdf/1804.08617.pdf]

2. Implementing Rainbow Algorithm [https://arxiv.org/pdf/1710.02298.pdf] which combines good features from different algorithms to form an itegrated agent.

3. Implementing parallel computing so that it is able to use multiple agents to train simultaneously. 
