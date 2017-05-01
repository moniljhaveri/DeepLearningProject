import tensorflow as tf 
import os 
import time 
import numpy as np 
import warnings 
from keras.layers import Conv2D, Flatten, Dense, Input 
from keras.models import Model  
warnings.filterwarnings('ignore')


def build_model(num_actions, channels, h, w, fc3_size=256): 
	state = tf.placeholder('float32', shape=[None, channels, h, w], name='State') 
	inputs = Input(shape=(channels, h, w,))
	model = Conv2D(16, (8, 8), activation='relu', border_mode='same', dim_ordering='tf')(inputs)
	model = Conv2D(32, (4,4),  activation='relu', border_mode='same', dim_ordering='tf')(model)
	model = Flatten()(model)
	model = Dense(output_dim=fc3_size, activation='relu')(model)
	out = Dense(output_dim=num_actions, activation='linear')(model)
	model = Model(input=inputs, output=out)
	return state, model

state, model = build_model(3, 84, 84, 3, 256) 
print("model built 2")

