import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # testing on tensorflow 1
import time


class ReplayExp():
    def __init__(self, N):
        self.buffer = deque(maxlen = N)
        
    def add(self, experience):
        self.buffer.append(experience)
    
    # take a random sample of k tuples of experience
    def sample_exp(self, batch_size):
        sample = random.choices(self.buffer, k = min(len(self.buffer), batch_size))
        
        return map(list, zip(*sample)) # return as a tuple of list
        

#Deep Q-Network
class DQN():
    # @params state_dim: dimension of each state --> NN input
    # @params action_size: dimension of each action --> NN output
    def __init__(self, state_dim, action_size): 
        #input current state
        self.state_in = tf.placeholder(tf.float32, shape = [None, *state_dim]) #None represents the batch size
        # current action a
        self.action_in = tf.placeholder(tf.int32, shape = [None]) #batch size
        
        # current estimate of Q-target
        self.q_target_in = tf.placeholder(tf.float32, shape = [None]) #batch size
        # encode actions in one-hot vector
        action_one_hot = tf.one_hot(self.action_in, depth = action_size)
        
        # hidden layer
        self.hidden1 = tf.layers.dense(self.state_in, 150, activation = tf.nn.relu) 
        self.hidden2 = tf.layers.dense(self.hidden1, 120, activation = tf.nn.relu) 
        
        # output Q_hat
        self.qhat = tf.layers.dense(self.hidden2, action_size, activation = None)
        
        # Q values of states and their corresponding actions a for each state
        # discard all non-taken actions
        self.qhat_s_a = tf.reduce_sum(tf.multiply(self.qhat, action_one_hot), axis = 1)
        # optimization objective
        self.loss = tf.reduce_mean(tf.square(self.q_target_in - self.qhat_s_a)) #mean of batch square error
        
        # We choose Adaptive Momentum as our optimization gradient descent
        self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        
    # update NN so that it estimate Q(s,a) closer to the target
    def update_nn(self, session, state, action, q_target):
        feed_info = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict = feed_info)
        
    def get_qhat(self, session, state):
        return session.run(self.qhat, feed_dict = {self.state_in: state}) # fill the placeholder


#The learning AI agent
class agent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape #state dimension
        self.action_size = env.action_space.n        # discrete action space (left or right)
        
        # the agent's "brain", tell the agent what to do
        self.brain = DQN(self.state_dim, self.action_size)
        
        self.epsilon = 1.0 # exploring prob to avoid local optima
        self.gamma = 0.99
        
        self.replay_exp = ReplayExp(N = 1000000)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def end_session(self):
        self.sess.close()
    
    def get_action(self, state): 
        qhat = self.brain.get_qhat(self.sess, [state]) # to match the placeholder dimenstion [None, *state_dim]
        prob = np.random.uniform(0.0, 1.0)
        
        if(prob < self.epsilon): # exploration
            action = np.random.randint(self.action_size)
        else: # exploitation
            action = np.argmax(qhat)
        return action
    
    def train(self, state, action, next_state, reward, done):
        # add exp to replay exp
        self.replay_exp.add((state, action, next_state, reward, done))
        
        states, actions, next_states, rewards, dones = self.replay_exp.sample_exp(batch_size = 80)
        
        # Q(s', _) --> Q-values for next state
        qhats_next = self.brain.get_qhat(self.sess, next_states)
        
        # set all value actions of terminal state to 0
        qhats_next[dones] = np.zeros((self.action_size))
            
        q_targets = rewards + self.gamma * np.max(qhats_next, axis=1) # update greedily
        
        self.brain.update_nn(self.sess, states, actions, q_targets)
        
        if done:
            self.epsilon = max(0.1, 0.98 * self.epsilon) #decaying exploration factor after each episode
        
        
        
    
env = gym.make("LunarLander-v2")
agent = agent(env)
num_episodes = 1000 #number of games

for episode in range(num_episodes):
    state = env.reset() # reset starting state for each new episode
    done = False
    reward_total = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        env.render() # display the environment after each action
        
        #base on the feedback from the environment, learn to update parameters for Q-value
        agent.train(state, action, next_state, reward, done)
        
        reward_total += reward
        state = next_state
        
        
    print("Episode number: ", episode, ", total reward:", reward_total)
    
    time.sleep(1)

# test after training
agent.epsilon = 0.0 # stop exploring
for episode in range(100):      
    state = env.reset() # reset starting state for each new episode
    done = False
  
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        env.render()
        state = next_state
        
    time.sleep(1)
        

