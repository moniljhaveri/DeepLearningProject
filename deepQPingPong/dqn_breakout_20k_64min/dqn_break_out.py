
# coding: utf-8

# In[7]:

import gym 
import numpy as np 
import tensorflow as tf 
#import keras 
from PIL import Image 
from collections import deque
import random 
# from keras import initializers
# from keras.initializers import normal, identity
# from keras.models import model_from_json
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.optimizers import SGD , Adam, RMSprop
import tensorflow as tf
import json
import skimage
from skimage import color 
from skimage import color, transform, exposure
import argparse
from collections import deque
import warnings
import pickle
warnings.filterwarnings('ignore')


# In[8]:

EPISODES = 20000
CONFIG = 'nothreshold'
 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
MAX_MEMORY = 100000
batch_sizes = 32
rewardList = []


# In[9]:

def preprocess_observation(self, observation): 
    x_t1 = color.rgb2gray(observation)
    x_t1 = skimage.transform.resize(x_t1, (80,80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0,255))
    x_t1 = np.uint8(x_t1)
        #x_t1 = x_t1.reshape(1,1, x_t1.shape[0], x_t1.shape[1])
    return x_t1 


# In[14]:

class DQNAgent(): 
    def __init__(self, state_size, action_size):
        with tf.device('/gpu:1'):
            
            self.state_size = state_size
            self.action_size = action_size 
            self.memory = deque(maxlen=1000000)
            self.gamma = 0.9 
            self.epsilon = 1.0 
            self.e_decay = 0.99 
            self.e_min = 0.05 
            self.learning_rate = 0.01 
            self.total_reward = 0.0
            self.total_loss = 0.0 
            self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

            self.state = tf.placeholder(tf.float32, [None, 80, 80, 4], name="state")
            self.action = tf.placeholder(tf.int32, shape=[None], name="actions")
            
            backToState = tf.to_float(self.state) / 255.0
            conv1 = tf.contrib.layers.conv2d(self.state, 32, 8, 8, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 4, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 3, activation_fn=tf.nn.relu)
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
            self.prediction = tf.contrib.layers.fully_connected(fc1, self.action_size)
            action_ind = tf.range(batch_sizes) * tf.shape(self.prediction)[1] + self.action
            self.Q = tf.gather(tf.reshape(self.prediction, [-1]), action_ind)

            # mse loss 
            self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.Q))
            self.optimizer =  tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def remember(self, state, action, reward, next_state, done):
        reward = np.clip(reward, -1.0, 1.0)
        self.total_reward += reward
        self.memory.append((state, action, reward, next_state, done))
        if(len(self.memory) > MAX_MEMORY): 
            self.memory.popleft()
            
    def act(self, sess, state): 
        if np.random.rand() <= self.epsilon: 
            return random.randrange(self.action_size)
        else: 
            state = state.reshape(1, state.shape[0], state.shape[1],state.shape[2])

            q = sess.run([self.prediction], {self.state: state})
            action = np.argmax(q)
            return action 

    def predict(self, sess, state):
            return sess.run([self.prediction], {self.state: state})

    def update(self, sess, state, action, y):
        feed_dict = {self.state: state, self.y: y, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
    
    def replay(self, sess, batch_size): 
        batch_size = min(batch_size, len(self.memory))
        
        minibatch = random.sample(self.memory, batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []
        
        for i in range(batch_size): 
            state, action, reward, next_state, done = minibatch[i]
            #target = reward
            #self.total_reward += reward
            
            if done: 
                #target = reward  
                terminal_batch.append(0)
            else: 
                terminal_batch.append(1)
            next_state_batch.append(next_state)
            action_batch.append(action)
            reward_batch.append(reward)
            state_batch.append(state) 
            #print("target ", target.shape)
            #print("targetf ",Y[i,:])
        terminal_batch = np.array(terminal_batch) + 0

        target_values = self.predict(sess, next_state_batch)[0]
        y_batch = reward_batch + self.gamma * (1 - terminal_batch) * np.max(target_values, axis=1)

        loss = self.update(sess, state_batch, action_batch, y_batch)
        self.total_loss += loss
        
        if self.epsilon > self.e_min: 
            self.epsilon *= self.e_decay 
            
    def return_loss():
        return self.total_loss
    def return_reward(): 
        return self.total_reward
      
    def preprocess_observation(self, observation): 
        x_t1 = color.rgb2gray(observation)
        x_t1 = skimage.transform.resize(x_t1, (80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0,255))
        x_t1 = np.uint8(x_t1)
        #x_t1 = x_t1.reshape(1,1, x_t1.shape[0], x_t1.shape[1])
        return x_t1 
     
  
    def initial_state(self, state):
        x_t = self.preprocess_observation(state)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) 
        s_t = s_t.reshape(s_t.shape[0], s_t.shape[1],s_t.shape[2])
        return s_t
    
    def set_reward(self, reward):
        self.total_reward = reward
        
    def load(self, name): 
        saver = tf.train.Saver()
        saver.restore(sess, name)

    def save(self, name): 
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), name)

    def set_loss(self):
        self.total_loss = 0


# In[ ]:




# In[ ]:

def main():
    env = gym.make('Breakout-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n 
    state = 0 
    config = tf.ConfigProto(allow_soft_placement = True)

    g = tf.Graph()
    with g.as_default(), tf.Session(config=config) as sess, tf.device('/gpu:1'):
        agent = DQNAgent(state_size, action_size)
        sess.run(tf.global_variables_initializer())
        
        for e in range(EPISODES):     
        #state = agent.preprocess_observation(state)          
            state = env.reset() 
            state = agent.initial_state(state)
            agent.set_reward(0)
            agent.set_loss()
        #state = agent.preprocess_observation(state)
            done = False 
            while not done: 
                #env.render() 
                action = agent.act(sess, state)
                next_state, reward, done, _ = env.step(action)
                next_state = agent.preprocess_observation(next_state) 
                next_state = next_state.reshape(next_state.shape[0], next_state.shape[1], 1)
                next_state = np.append(next_state, state[:,:,:3], axis=2)
                reward = reward if not done else -10 
                agent.remember(state, action, reward, next_state, done)
                state = next_state 
                if (e > 0):
                    agent.replay(sess, 32)
                    if done: 
                        output += "\n" + "episode: {}/{}, score: {}, loss: {}".format(e, EPISODES, agent.total_reward, agent.loss)
                        rewardList.append(agent.total_reward)

                        if (e%10 == 0):
                            agent.save('model1')
                            with open("test.txt", "a") as myfile:
                                myfile.write(output)
                            pickle.dump(rewardList, open('rewards.p', 'wb'))
                            output = " "
                        break
                    #print(sumReward)         
                else: 
                    if done:
                        output = "\n" "episode: {}/{}, score: {}, loss: {}".format(e, EPISODES, agent.total_reward, agent.loss)
                        with open("test.txt", "a") as myfile:
                            myfile.write(output)
                        rewardList.append(agent.total_reward)
                        pickle.dump(rewardList, open('rewards.p', 'wb'))
                        print(output)
                        break 
            
        

    env.close()

main()


# In[ ]:

env = gym.make('Pong-v0')

env.close()


# In[ ]:

env.close()


# In[ ]:



