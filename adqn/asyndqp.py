import os 
import threading 
import tensorflow as tf 
import sys 
import numpy as np 
import time 
import gym 
import random
os.environ["KERAS_BACKEND"] = "tensorflow" 

from skimage.transform import resize
from skimage.color import rgb2gray
from atari_enviroment import AtariEnvironment
from build_model import build_model 
from keras import backend as K 

flags = tf.app.flags

flags.DEFINE_string('experiment', 'async_dqn_pong', 'Name of the current experiment')
flags.DEFINE_string('game', 'Pong-v0', 'Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent actor-learner threads to use during training.')
flags.DEFINE_integer('tmax', 80000000, 'Number of training timesteps.')
flags.DEFINE_integer('resized_width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('resized_height', 84, 'Scale screen to this height.')
flags.DEFINE_integer('agent_history_length', 4, 'Use this number of recent screens as the environment state.')
flags.DEFINE_integer('network_update_frequency', 32, 'Frequency with which each actor learner thread does an async gradient update')
flags.DEFINE_integer('target_network_update_frequency', 10000, 'Reset the target network every n timesteps')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, 'Number of timesteps to anneal epsilon.')
flags.DEFINE_string('summary_dir', '/tmp/summaries', 'Directory for storing tensorboard summaries')
flags.DEFINE_string('checkpoint_dir', '/tmp/checkpoints', 'Directory for storing model checkpoints')
flags.DEFINE_integer('summary_interval', 5,
                     'Save training summary to file every n seconds (rounded '
                     'up to statistics interval.')
flags.DEFINE_integer('checkpoint_interval', 500,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'seconds (rounded up to statistics interval.')
flags.DEFINE_boolean('show_training', True, 'If true, have gym render evironments during training')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_string('checkpoint_path', 'tmp/checkpoints/recent.ckpt', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_string('eval_dir', '/tmp/', 'Directory to store gym evaluation')
flags.DEFINE_integer('num_eval_episodes', 100, 'Number of episodes to run gym evaluation.')
FLAGS = flags.FLAGS
T = 0
TMAX = FLAGS.tmax

def sample_final_epsilon(): 
	final_eps = np.array([.1, 0.1, .5])
	prob = np.array([.4, 0.3, .3])
	return np.random.choice(final_eps, 1, p=list(prob))[0]

def actor_learner_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver, summary_writer): 
	global TMAX, T 
	s = graph_ops["s"] 
	q_values = graph_ops['q_values']
	st = graph_ops['st'] 
	target_q_values = graph_ops['target_q_values'] 
	a = graph_ops['a'] 
	y = graph_ops['y'] 
	grad_update = graph_ops['grad_update'] 

	summary_placeholders, update_ops, summary_op = summary_ops

	env = AtariEnvironment(gym_env=env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, agent_history_length=FLAGS.agent_history_length)

	s_batch = []
	a_batch = []
	y_batch = [] 

	final_epsilon = sample_final_epsilon() 
	initial_epsilon = 1.0 
	epsilon = 1.0 
	print("Starting thread ", thread_id, "with final epsilon ", final_epsilon)
	
	time.sleep(3*thread_id)
	t = 0 	
	while T < TMAX: 
		s_t = env.get_initial_state() 
		terminal = False 

		ep_reward = 0 
		episode_ave_max_q = 0 
		ep_t = 0 
		
		while True: 
			readout_t = q_values.eval(session = session, feed_dict={s :[s_t]})
			a_t = np.zeros([num_actions]) 
			action_index = 0 
			if random.random() <= epsilon: 
				action_index = random.randrange(num_actions)
			else: 
				action_index  = np.argmax(readout_t)
			a_t[action_index] = 1 
			
			if epsilon > final_epsilon: 
				epsilon -= (initial_epsilon - final_epsilon)/FLAGS.anneal_epsilon_timesteps

			s_t1, r_t, terminal, info = env.step(action_index)
			
			readout_j1 = target_q_values.eval(session = session, feed_dict={st : [s_t1]})
			clipped_r_t = np.clip(r_t, -1, 1) 
			
			if terminal: 
				y_batch.append(clipped_r_t)
			else: 
				y_batch.append(clipped_r_t + FLAGS.gamma * np.max(readout_j1))
			
			a_batch.append(a_t) 
			s_batch.append(s_t) 

			s_t = s_t1 
			T += 1 
			t += 1 
			
			ep_t += 1 
			ep_reward += r_t 

			episode_ave_max_q += np.max(readout_t) 

			if T%FLAGS.target_network_update_frequency == 0: 
				session.run(reset_target_network_params)
			
			if T % FLAGS.network_update_frequency == 0 or terminal: 
				if s_batch: 
					session.run(grad_update, feed_dict = {y: y_batch, a: a_batch, s: s_batch}) 
				s_batch = [] 
				a_batch = [] 
				y_batch = [] 
			if t % FLAGS.checkpoint_interval == 0: 
				saver.save(session, FLAGS.checkpoint_dir + "/" + FLAGS.experiment+".ckpt", global_step = t) 	
				
			if terminal: 
				stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon] 
				for i in range(len(stats)): 
					session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
					data = session.run(summary_ops)
					summary_str = session.run(summary_op)
					summary_writer.add_summary(summary_str, float(T))
				print("Thread: ", thread_id, "/ Time", T, "/ TIMESTEP ", t, "/ EPSILON ", epsilon, "/ REWARD ", ep_reward, "/ Q_MAX %.4f" % (episode_ave_max_q/float(ep_t)), "/ EPSILON PROGRESS ", t/float(FLAGS.anneal_epsilon_timesteps)) 
				break 	
def setup_summaries():
	episode_reward = tf.Variable(0.)
	tf.summary.scalar("Episode Reward", episode_reward)
	episode_ave_max_q = tf.Variable(0.)
	tf.summary.scalar("Max Q Value", episode_ave_max_q)
	logged_epsilon = tf.Variable(0.)
	tf.summary.scalar("Epsilon", logged_epsilon)
	logged_T = tf.Variable(0.)
	summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
	summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
	update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
	summary_op = tf.summary.merge_all()
	return summary_placeholders, update_ops, summary_op

def build_graph(num_actions): 
	s, q_network = build_model(num_actions, FLAGS.agent_history_length, FLAGS.resized_width, FLAGS.resized_height)	
	network_params = q_network.trainable_weights
	q_values = q_network(s) 

	st, target_q_network = build_model(num_actions, FLAGS.agent_history_length, FLAGS.resized_width, FLAGS.resized_height)	
	
	target_network_params = target_q_network.trainable_weights
	target_q_values = target_q_network(st)
	
	reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
	print("here")

	a = tf.placeholder("float32", [None, num_actions]) 
	y  = tf.placeholder("float32", [None]) 

	action_q_values = tf.reduce_sum(q_values * a, reduction_indices=1)
	loss = tf.reduce_mean(tf.square(y - action_q_values)) 
	optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
	grad_update = optimizer.minimize(loss, var_list=network_params)

	graph_ops = {'s': s, 'q_values':q_values, 'st': st, 'a' : a, 'y':y,'target_q_values':target_q_values, 'reset_target_network_params': reset_target_network_params, 'grad_update':grad_update} 
	print("built graphs") 
	return graph_ops 

def get_num_actions(): 
	env = gym.make(FLAGS.game)
	num_actions = env.action_space.n 
	if(FLAGS.game == 'Pong-v0' or FLAGS.game == 'Breakout-v0'): 
		num_actions = 3 
	return num_actions 

def train(session, graph_ops, num_actions, saver): 
	session.run(tf.global_variables_initializer())
	session.run(graph_ops['reset_target_network_params']) 
	envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)] 
	summary_ops = setup_summaries() 
	summary_op = summary_ops[-1]
	summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment
	summary_writer = tf.summary.FileWriter(summary_save_path, graph=tf.get_default_graph())
	
	if not os.path.exists(FLAGS.checkpoint_dir): 
		os.makedirs(FLAGS.checkpoint_dir) 

	actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id, envs[thread_id], session, graph_ops, num_actions, summary_ops, saver, summary_writer)) for thread_id in range(FLAGS.num_concurrent)]
	for t in actor_learner_threads: 
		t.start() 

	last_summary_time = 0 
	while True: 
		for env in envs: 
			env.render() 
		now = time.time() 
		if now - last_summary_time > FLAGS.summary_interval: 
			#summary_str = session.run(summary_op)
			#summary_writer.add_summary(summary_str, float(T))
			last_summary_time = now 
	
	for t in actor_learner_threads: 
		t.join() 	
		print("train")

def main(_): 
	g = tf.Graph() 
	with g.as_default(), tf.Session() as session: 
		K.set_session(session)
		num_actions = get_num_actions()
		graph_ops = build_graph(num_actions)
		saver = tf.train.Saver() 
		train(session, graph_ops, num_actions, saver)
if __name__ == "__main__":	
	tf.app.run()
