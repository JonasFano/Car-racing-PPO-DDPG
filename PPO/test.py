from env import Env
from agent import Agent
import numpy as np
import torch
from config import Config


def main():
    """
    - Check if GPU is available and set device accordingly
    - Initialize agent and load parameters
    - Initialize environment
    - Loop over a fixed number of episodes
        - Reset the environment to get the initial state
        - Loop over steps within the episode
            - Select an action using the agent's policy
            - Perform a step in the environment with the selected action
            - Update the episode score
            - Check for termination conditions
        - Print episode information
    """

    # Check if GPU is available and set device accordingly
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set random seed for reproducibility
    torch.manual_seed(Config.seed)
    if use_cuda:
        torch.cuda.manual_seed(Config.seed)

    # Initialize agent and load parameters
    agent = Agent(device, training=False)
    agent.load_model()
    
    # Initialize environment
    environment = Env()

    # Loop over a fixed number of episodes
    for episode in range(50):
        episode_reward = 0
        state = environment.reset()
        

        # Loop over steps within the episode
        for t in range(1000):
            # Select an action using the agent's policy
            action = agent.select_action(state)
            # Perform a step in the environment with the selected action
            # The action is being applied Config.action_repeat times
            next_state, reward, done, _, die = environment.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            # Update the episode reward
            episode_reward += reward
            state = next_state

            if done or die:
                break
        
        # Print episode information
        print('Ep {}\tScore: {:.2f}\t'.format(episode, episode_reward))

if __name__ == "__main__":
    main()