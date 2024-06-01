######################################################################################################
# Config Class for defining all arguments/hyperparameters in one file

# Most implementation and parameter choices are based on the implementation of Ma, X. 
# The corresponding link to the GitHub repository: 
# https://github.com/xtma/pytorch_car_caring/tree/master (Link last updated on 01/06/2024).
######################################################################################################

class Config:
    # Hyperparameters for Reinforcement Learning (PPO)
    gamma = 0.99                # Discount factor for future rewards
    action_repeat = 8           # Number of times to repeat each action before the next action is taken
    img_stack = 4               # Number of consecutive frames stacked as input
    buffer_capacity = 2000      # Maximum capacity of the replay buffer
    batch_size = 128            # Batch size for mini-batch gradient descent
    epoch_count = 10            # Number of epochs for updating the policy
    learning_rate = 1e-3        # Learning rate for the optimizer
    ema_smooth_factor = 0.1     # Smoothing factor for calculating exponential moving average (EMA) during training

    # Environment and Visualization Settings
    seed = 0                        # Seed value for random number generation
    vis = True                      # Whether to visualize the environment
                                    # Before running the code with "vis = True" start the visdom server in the command prompt
                                    # python -m visdom.server
    render_mode = "human"    # Mode for rendering the environment
                                    # Options: "human", "rgb_array", "state_pixels"

    # File Names and Logging
    train_model_name = "ppo_net_params_v5.pkl"  # File name for saving the trained model parameters
    test_model_name = "ppo_net_params_v1.pkl"   # File name for loading the trained model parameters for testing
    csv_log_file = "training_data_v5.csv"       # File name of a csv file for logging training data
                                                # v4 - use Gymnasium's intrinsic reward structure (penalize die and don't penalize green)
                                                # v3 - don't penalize die
                                                # v2 - penalize green with additional -0.1 reward
                                                # v1 - use optimized reward structure 

    # Logging Interval and settings
    log_interval = 10           # Interval for logging training statistics
    save = True                 # Whether to save the trained model