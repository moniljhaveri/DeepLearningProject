# This class implements an Ornstein-Uhlenbeck random noise generator.  The noise will be added to the policy to enable
# the agent to experiment with different actions.

# Reference: https://github.com/stevenpjg/ddpg-aigym/blob/master/ou_noise.py

import numpy as np

class OUNoise:

    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.45):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    # This function willl be used to get the next noise to add to every step
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
