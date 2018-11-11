# Project 2 : Continuous Control

## Project Details:

For this project, an agent is trained to control a double-jointed arm to move to target locations  

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

## Getting Started:

To run the code, you need to have PC Windows (64-bit) with Anaconda with Python 3.6 installed.

To download Anaconda, please click the link below:

https://www.anaconda.com/download/

Clone or download and unzip the DRLND_P2_Continuous_Control folder.

Download by clicking the link below and unzip the environment file under DRLND_P2_Continuous_Control folder

https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip

Download by clicking the link below and unzip the ml-agents file under DRLND_P2_Continuous_Control folder

https://github.com/Unity-Technologies/ml-agents/tree/0.4.0b

### Dependencies :

To set up your Python environment to run the code in this repository, follow the instructions below.

  1. Create (and activate) a new environment with Python 3.6.
  
 ```
 conda create --name drlnd python=3.6
 activate drlnd
 ```
  2. Install Pytorch by following the instructions in the link below.
  
     https://pytorch.org/get-started/locally/
    
  3. Then navigate to P1_Navigation/ml-agents-0.4.0b/python and install ml-agent.
     ```
     pip install .
     ```
  4. Install matplotlib for plotting graphs.
     ```
     conda install -c conda-forge matplotlib
     ```
  5. (Optional) Install latest prompt_toolkit may help in case Jupyter Kernel keeps dying
     ```
     conda install -c anaconda prompt_toolkit 
     ```
     
## Run the code 

  Open Navigation.ipynb in Jupyter and press Ctrl+Enter to run the first cell to import all the libraries.
  
  ### Watch a random agent
   Run 2nd cell  to 6th cell to watch how a random agent plays.
   
  ### Train an agent
   Run the cells which contain the "train" function and then choose those cells with "train an agent with ..." in comment
   
  ### Watch a trained agent
   Run the cells with "watch a ... trained agent" to watch how an agent trained by a particular algorithm behaves.


