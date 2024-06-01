import torch.nn as nn
from config import Config


class Net(nn.Module):
    """
    Actor-Critic Network for the PPO algorithm
    """

    def __init__(self):
        """
        Creates the Actor-Critic Convolutional Neural Network for PPO
        Network is directly based on the implementation of xtma:
        https://github.com/xtma/pytorch_car_caring/tree/master
        """
        super(Net, self).__init__()
        # Convolutional base layers
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(Config.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        
        # Value/critic head to estimate the state value function
        self.v = nn.Sequential(
            nn.Linear(256, 100), 
            nn.ReLU(), 
            nn.Linear(100, 1))
        
        # Action/Actor head to get the parameters of a Beta distribution
        self.fc = nn.Sequential(
            nn.Linear(256, 100), 
            nn.ReLU())
        self.alpha_head = nn.Sequential(
            nn.Linear(100, 3), 
            nn.Softplus())
        self.beta_head = nn.Sequential(
            nn.Linear(100, 3), 
            nn.Softplus())

        # Initialize weights
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        """
        Custom weight initialization function.

        Parameters:
        - m (torch.nn.Module): Module to initialize weights.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - (alpha, beta) (tuple): Tuple containing alpha and beta parameters.
                - alpha: Shape parameter that determines the concentration of 
                        probability around the lower bound of the probability 
                        distribution of PPO.
                - beta: Shape parameter that determines the concentration of 
                        probability around the upper bound of the probability 
                        distribution of PPO.
        - v (torch.Tensor): Value prediction.
        """
        cnn_out = self.cnn_base(x)
        cnn_out = cnn_out.view(-1, 256)
        v = self.v(cnn_out)
        cnn_out = self.fc(cnn_out)
        alpha = self.alpha_head(cnn_out) + 1
        beta = self.beta_head(cnn_out) + 1

        return (alpha, beta), v