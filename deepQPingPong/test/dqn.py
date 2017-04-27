
# coding: utf-8

# In[5]:

import gym 
import numpy as np 
import tensorflow as tf 
import keras 
from PIL import Image 
from collections import deque
import random 
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam, RMSprop
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


# In[6]:

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
rewardList = []


# In[7]:

class DQNAgent: 
    def __init__(self, state_size, action_size): 
        self.state_size = state_size
        self.action_size = action_size 
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9 
        self.epsilon = 1.0 
        self.e_decay = 0.99 
        self.e_min = 0.05 
        self.learning_rate = 0.01 
        self.model = self._build_model()
        self.loss = 0.0
        self.total_reward = 0.0
    def _build_model(self): 
        with tf.device('cpu:0'):
            model = Sequential()
            model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(80,80,4)))  #80*80*4
            model.add(Activation('relu'))
            model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
            model.add(Activation('relu'))
            model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dense(self.action_size))
   
            adam = Adam(lr=LEARNING_RATE)
            model.compile(loss='mse',optimizer=adam)
        return model 
    
    def set_loss(self): 
        self.loss = 0
    
    def remember(self, state, action, reward, next_state, done):
        reward = np.clip(reward, -1.0, 1.0)
        self.total_reward += reward
        self.memory.append((state, action, reward, next_state, done))
        if(len(self.memory) > MAX_MEMORY): 
            self.memory.popleft()
    
    def act(self, state): 
        if np.random.rand() <= self.epsilon: 
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size): 
        batch_size = min(batch_size, len(self.memory))
        
        minibatch = random.sample(self.memory, batch_size) 
        X = np.zeros((batch_size, 80, 80, 4))
        Y = np.zeros((batch_size, self.action_size)) 

        for i in range(batch_size): 
            state, action, reward, next_state, done = minibatch[i]
            Q_sa = self.model.predict(state)
            target = reward
            #self.total_reward += reward
            if done: 
                target = reward  
            else: 
                target = reward + self.gamma * np.max(Q_sa)
            X[i], Y[i] = state, target
            #print("target ", target.shape)
            #print("targetf ",Y[i,:])
        history = self.model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0)
        self.loss += history.history['loss'][0]
        if self.epsilon > self.e_min: 
            self.epsilon *= self.e_decay 
            
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
        s_t = s_t.reshape(1,  s_t.shape[0], s_t.shape[1],s_t.shape[2])
        return s_t
    
    def set_reward(self, reward):
        self.total_reward = reward
        
    def load(self, name): 
        self.model.save_weights(name)

    def save(self, name): 
        self.model.save_weights(name)


# In[ ]:




# In[ ]:

if __name__ == "__main__": 
    env = gym.make('Pong-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n 
    agent = DQNAgent(state_size, action_size)
    #state = env.reset() 
    output = []
    for e in range(EPISODES):     
        #state = agent.preprocess_observation(state)  
        
        sumReward = 0 
        
        state = env.reset() 
        state = agent.initial_state(state)
        agent.set_reward(0)
        agent.set_loss()
        #state = agent.preprocess_observation(state)
        done = False 
        while not done: 
            #env.render() 
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = agent.preprocess_observation(next_state) 
            next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1], 1)
            next_state = np.append(next_state, state[:,:,:,:3], axis=3)

            reward = reward if not done else -10 
            sumReward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state 
            if (e > 0):
                agent.replay(32)
                if done: 
                    output += "\n" + "episode: {}/{}, score: {}, loss: {}".format(e, EPISODES, agent.total_reward, agent.loss)
                    rewardList.append(agent.total_reward)

                    if (e%1 == 0):
                        agent.save('model.h5')
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


# In[ ]:

env.close()


# In[ ]:



