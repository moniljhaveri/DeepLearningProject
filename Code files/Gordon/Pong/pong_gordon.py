import keras

import numpy as np
import gym
import tensorflow as tf

# hyperparameters
n_obs = 80 * 80           # dimensionality of observations
h = 200                   # number of hidden layer neurons
n_actions = 3             # number of available actions
learning_rate = 1e-3
gamma = .99               # discount factor for reward
decay = 0.99              # decay rate for RMSProp gradients
save_path='models-TF/pong.ckpt'

# gamespace 
env = gym.make("Pong-v0") # environment info
observation = env.reset()
prev_x = None
xs,rs,ys = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 0











