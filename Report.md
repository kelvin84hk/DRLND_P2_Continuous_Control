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


