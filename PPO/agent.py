import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from model import Net
from config import Config


class Agent():
    """
    Agent for training and testing depending on the 'training' variable.
    """
    def __init__(self, device, training = False):
        self.training = training
        self.epoch_count = Config.epoch_count
        self.buffer_capacity = Config.buffer_capacity
        self.batch_size = Config.batch_size
        self.gamma = Config.gamma
        self.device = device

         # Define the buffer data type for storing data collected during training
        self.buffer_dtype = np.dtype([('state', np.float64, (Config.img_stack, 96, 96)), ('next_state', np.float64, (Config.img_stack, 96, 96)), 
                                    ('action_log_prob', np.float64), ('action', np.float64, (3,)), ('reward', np.float64)])

        if self.training:
            self.net = Net().to(self.device).double()
            self.buffer = np.empty(self.buffer_capacity, dtype = self.buffer_dtype)
            self.store_counter = 0
            self.optimizer = optim.Adam(self.net.parameters(), lr = Config.learning_rate)
        else:
            self.net = Net().to(self.device).double()


    def select_action(self, state):
        """
        Selects an action based on the given state.

        Parameters:
        - state (np.array): Input state.

        Returns:
        - action (np.array): Selected action.
        - action_log_prob (float): Log probability of the selected action.
        """
        if self.training:
            state = torch.from_numpy(state).to(self.device).double().unsqueeze(0)
            with torch.no_grad():
                alpha, beta = self.net(state)[0]
            dist = Beta(alpha, beta)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=1)

            action = action.squeeze().cpu().numpy()
            action_log_prob = action_log_prob.item()
            return action, action_log_prob
        else:
            state = torch.from_numpy(state).to(self.device).double().unsqueeze(0)
            with torch.no_grad():
                alpha, beta = self.net(state)[0]
            action = alpha / (alpha + beta)

            action = action.squeeze().cpu().numpy()
            return action
        

    def update(self):
        """
        Performs parameter update based on stored data using the Proximal Policy Optimization (PPO) algorithm.
        """
        # Prepare tensors for states, actions, rewards, next states, and old action log probabilities
        state = torch.tensor(self.buffer['state'], dtype = torch.double).to(self.device)
        action = torch.tensor(self.buffer['action'], dtype = torch.double).to(self.device)
        reward = torch.tensor(self.buffer['reward'], dtype = torch.double).to(self.device).view(-1, 1)
        next_state = torch.tensor(self.buffer['next_state'], dtype  =torch.double).to(self.device)
        old_action_log_prob = torch.tensor(self.buffer['action_log_prob'], dtype = torch.double).to(self.device).view(-1, 1)

        # Calculate advantage using the formula: advantage = v_target - self.net(state)[1]
        with torch.no_grad():
            v_target = reward + self.gamma * self.net(next_state)[1] # V(s_(t+1))
            advantage = v_target - self.net(state)[1] # V(s_t)

        # Iterate over the stored transition data for a certain number of epochs
        for _ in range(self.epoch_count):
            # Sample batches of transitions using BatchSampler and SubsetRandomSampler
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                # Calculate action log probabilities for the sampled actions using the current policy network
                alpha, beta = self.net(state[index])[0]
                dist = Beta(alpha, beta)
                action_log_prob = dist.log_prob(action[index]).sum(dim = 1, keepdim = True)

                # Calculate the ratio and surrogate objectives for policy optimization
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                surrogate1 = ratio * advantage[index]
                epsilon = 0.2
                surrogate2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage[index]

                # Calculate the policy/actor loss using the surrogate
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Calculate the value/critic loss using smooth L1 loss between predicted value and target value
                critic_loss = F.smooth_l1_loss(self.net(state[index])[1], v_target[index])

                # Compute the total loss as the sum of policy loss and twice the value loss
                loss = actor_loss + 2. * critic_loss

                # Perform backpropagation through the total loss and update parameters using Adam optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def save_model(self):
        """
        Saves the model parameters.
        """
        torch.save(self.net.state_dict(), 'models/' + Config.train_model_name)


    def load_model(self):
        """
        Loads the model parameters.
        """
        self.net.load_state_dict(torch.load('models/' + Config.test_model_name, map_location = self.device))


    def store(self, buffer_data):
        """
        Stores transition data in the buffer.

        Parameters:
        - buffer_data (tuple): Transition data.

        Returns:
        - bool: True if buffer is full, False otherwise.
        """
        self.buffer[self.store_counter] = buffer_data
        self.store_counter += 1
        # If buffer is full, start my overwriting oldest entry
        if self.store_counter == self.buffer_capacity:
            self.store_counter = 0
            return True
        return False