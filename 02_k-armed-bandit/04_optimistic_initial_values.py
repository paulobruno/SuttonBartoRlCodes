import numpy as np
import matplotlib.pyplot as plt
import random


k = 10 # num of bandits
mu = 0 # q*(a) average
sigma = 1 # q*(a) standard deviation

num_of_problems = 2000
time_steps = 1000
alpha = 0.1


def epsilon_greedy(q, epsilon):
    if (random.random() < epsilon):
        a = random.randint(0, len(q) - 1)
    else:
        a_array = np.nonzero(q == q[np.argmax(q)])
        a = random.choice(a_array[0])
    return a

def run_simulation(q_real, epsilon, initial_q):
    r_received = np.zeros((time_steps))
    optimal_action_taken = np.zeros((time_steps))

    optimal_a = np.argmax(q_real)

    q_estimate = np.full((k), initial_q, dtype=np.float64)
    #n = np.zeros((k))

    for step in range(time_steps):
        action = epsilon_greedy(q_estimate, epsilon)

        r = np.random.normal(q_real[action], sigma)

        #n[action] = n[action] + 1
        #q_estimate[action] = q_estimate[action] + ((r - q_estimate[action]) / n[action])
        q_estimate[action] = q_estimate[action] + alpha * (r - q_estimate[action])

        r_received[step] = r_received[step] + r
        optimal_action_taken[step] = optimal_action_taken[step] + (1 if action == optimal_a else 0)

    return r_received, optimal_action_taken

r_0_5 = np.zeros((time_steps))
r_10_0 = np.zeros((time_steps))

optimal_action_0_5 = np.zeros((time_steps))
optimal_action_10_0 = np.zeros((time_steps))

for problem in range(num_of_problems):
    bandit_q = np.random.normal(mu, sigma, size=(k))
    
    r, o = run_simulation(bandit_q, 0, 5)
    r_0_5 = np.add(r_0_5, r)
    optimal_action_0_5 = np.add(optimal_action_0_5, o)
    
    r, o = run_simulation(bandit_q, 0.1, 0)
    r_10_0 = np.add(r_10_0, r)
    optimal_action_10_0 = np.add(optimal_action_10_0, o)

plt.plot(r_0_5 / num_of_problems, color='b')
plt.plot(r_10_0 / num_of_problems, color='r')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.show()

plt.plot(optimal_action_0_5 / num_of_problems, color='b')
plt.plot(optimal_action_10_0 / num_of_problems, color='r')
plt.axis([0, time_steps, 0, 1.0])
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.show()