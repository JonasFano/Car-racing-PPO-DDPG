# Car_racing_PPO_DDPG

This project was done in the Tools of Artificial Intelligence course of the Master's programme "Robot Systems - Advanced Robotics Technology" at the University of Southern Denmark (SDU).

It includes a PPO algorithm optimized for training an reinforcement learning agent to navigate through the Gymnasium Car-Racing environment. 

Additionally, it uses DDPG from the Stable-Baselines3 library to train an agent for comparison.

The documentation of this project was submitted to the university on 01/06/2024 in form of a report.
\\
\\
This GitHub includes 4 pre-trained models. Each was trained approx. 2000 episodes with each 1000 timesteps. The parameters are the same except the reward structure:
- v4 - use Gymnasium's intrinsic reward structure (penalize die and don't penalize green)
- v3 - don't penalize die
- v2 - penalize green with additional -0.1 reward
- v1 - use optimized reward structure 
\\
\\
Description of the folders and files:
- DDPG: Folder with all files used for training the agent with DDPG and visualizing the results
  - main.ipynb: Jupyther Source File to train the agent with DDPG from the Stable-Baselines3 library
  - data.txt: Textfile containing the result data obtained during training
  - visu.py: Pythonfile for visualizing the results in the data.txt file 
- PPO: Folder containts the used PPO algorithm and files for visualizing results
  - 
  - 
