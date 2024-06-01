import numpy as np
import gymnasium as gym
from config import Config
import cv2


class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        """
        Initializes the environment.
        Creates an instance of the CarRacing environment with the specified render mode.
        """
        self.env = gym.make('CarRacing-v2', render_mode=Config.render_mode)
        self.reward_threshold = self.env.spec.reward_threshold
        self.history_reward = 0

    def step(self, action):
        """
        Performs a step in the environment given an action.

        Parameters:
        - action (int): The action to take in the environment.

        Returns:
        - stack (np.array): Next state after taking the action.
        - total_reward (float): Total reward obtained from taking the action.
        - done (bool): Whether the episode has ended.
        - _ (object): Placeholder.
        - die (bool): Whether the agent has "died" (reached the border of the map).
        """
        total_reward = 0
        for _ in range(Config.action_repeat): # Repeat Config.action_repeat times the given action
            img_rgb, reward, done, _, die = self.env.step(action)
            # if die:
            #     reward += 100
            # # green penalty
            # if np.mean(img_rgb[:, :, 1]) > 185.0:
            #     reward -= 0.05
            total_reward += reward

            # Check for early termination of the episode if reward is negative for a long time
            if reward < 0:
                self.history_reward += 1
                if self.history_reward > 80:
                    done = True
            else:
                self.history_reward = 0

            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        return np.array(self.stack), total_reward, done, _, die

    @staticmethod
    def rgb2gray(rgb):
        """
        Converts an RGB image to grayscale.

        Parameters:
        - rgb (np.array): RGB image array.

        Returns:
        - gray (np.array): Grayscale image array.
        """
        # RGB image -> Gray [0, 255]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Normalize to Gray [-1, 1]
        gray = cv2.normalize(gray, None, alpha=-1.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return gray

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
        - stack (np.array): Initial state after reset.
        """
        self.counter = 0
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb[0])
        self.stack = [img_gray] * Config.img_stack  # four frames for decision
        return np.array(self.stack)