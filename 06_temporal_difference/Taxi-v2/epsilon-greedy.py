# Importing the environment
import gym

env = gym.make('Taxi-v2').env

env.reset()
env.render()


# Training the agent
import random
import gym
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

training_episodes = 10000
learning_rate = 0.1
discount_factor = 0.9

epsilon = 0.1

q_table = np.zeros([env.observation_space.n, env.action_space.n])

steps_per_episode2 = np.zeros((training_episodes))
    
for i in trange(0, training_episodes):
    
    state = env.reset()

    done = False    
    
    num_steps = 0
    
    while not done:
        
        num_steps += 1
        
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)
        
        q_value = q_table[state, action]
        max_q = np.max(q_table[next_state])
        
        new_q = q_value + learning_rate * (reward + discount_factor * max_q - q_value)
        q_table[state, action] = new_q

        state = next_state        
        
    steps_per_episode2[i] = num_steps
        
print("Avg num of steps per episode: " + str(np.mean(steps_per_episode2)))


# Testing the agent
import time

from IPython.display import clear_output

test_episodes = 5

for i in range(test_episodes):
    
    state = env.reset()
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)

        clear_output(wait=True)
        env.render()
        
        time.sleep(1)
        
    print("Finished episode " + str(i) + "/" + str(test_episodes))
    time.sleep(3)