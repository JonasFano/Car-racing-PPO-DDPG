# Training code
# The full code is inspired by xtma
# https://github.com/xtma/pytorch_car_caring/tree/master


from env import Env
from agent import Agent
import numpy as np
import torch
from config import Config
import os
import csv


def main():
    """
    Performs training of the environment using a simplified version of the PPO algorithm.
    Steps that are being performed:
        - Initialize the agent and environment
        - Loop over episodes
            - Reset the environment to get the initial state
            - Loop over steps within the episode
                - Select an action using the agent's policy
                - Perform a step in the environment with the selected action
                - Store transition tuple and update agent's parameters if applicable
                - Update the episode reward
                - Check for termination conditions
            - Update the exponential moving avaerage (EMA) reward
            - Save agent's parameters and save/visualize training data
    """

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set random seed for reproducibility
    torch.manual_seed(Config.seed)
    if use_cuda:
        torch.cuda.manual_seed(Config.seed)

    # Initialize agent
    agent = Agent(device, training=True)

    # Initialize environment
    environment = Env()

    # Create log directory if it does not exist
    log_path = os.path.join('Training', 'Logs')
    os.makedirs(log_path, exist_ok=True)
    csv_file = os.path.join(log_path, Config.csv_log_file)
    if os.path.exists(csv_file):
        os.remove(csv_file)

    ############################
    #### Main Training Loop ####
    ############################
    # Initialize a list to store episode rewards for the last log interval episodes
    list_episode_rewards = []
    ema_reward = 0

    for episode in range(10000):
        episode_reward = 0
        state = environment.reset()

        for _ in range(1000):
            # Select an action using the agent's policy
            action, action_log_prob = agent.select_action(state)
            # Perform a step in the environment with the selected action
            # The action is being applied Config.action_repeat times
            next_state, reward, done, _, die = environment.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            
            # Store and update agent's parameters
            if agent.store((state, next_state, action_log_prob, action, reward)):
                print('updating')
                agent.update()

            # Update the episode reward
            episode_reward += reward
            state = next_state
            if done or die:
                break

        # Add the episode reward to the list of episode rewards
        list_episode_rewards.append(episode_reward)
        # If the list of episode rewards exceeds the log interval, remove the oldest reward
        if len(list_episode_rewards) > Config.log_interval:
            list_episode_rewards.pop(0)
        # Compute the average episode reward for the last log interval episodes
        interval_reward = np.mean(list_episode_rewards)
        
        # Update exponential moving average (EMA) reward
        ema_reward = Config.ema_smooth_factor * episode_reward + (1 - Config.ema_smooth_factor) * ema_reward

        if episode % Config.log_interval == 0:
            # Save episode data to CSV file if Config.save is True but only at regular intervals
            if Config.save:
                episode_data = {'Episode': episode, 'Episode reward': episode_reward, 'Interval reward': interval_reward, 'EMA reward': ema_reward}
                write_header = not os.path.exists(csv_file)
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames = episode_data.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(episode_data)

            print('Ep {}\tLast episode reward: {:.2f}\tAverage reward (last {} episodes): {:.2f}\tEMA reward: {:.2f}'.format(
                episode, episode_reward, len(list_episode_rewards), interval_reward, ema_reward))
            agent.save_model()
            
        if ema_reward > environment.reward_threshold:
            print("Training done!\tEp {}\tLast episode reward: {}\tEMA reward: {}!".format(episode, episode_reward, ema_reward))
            break


if __name__ == "__main__":
    main()
