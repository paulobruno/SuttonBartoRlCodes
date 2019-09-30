import numpy as np
import matplotlib.pyplot as plt
import random


k = 10 # num of bandits
mu = 0 # q*(a) average
sigma = 1 # q*(a) standard deviation

num_of_problems = 2000
time_steps = 10000
alpha = 0.1


def epsilon_greedy(q, epsilon):
    if (random.random() < epsilon):
        a = random.randint(0, len(q) - 1)
    else:
        a_array = np.nonzero(q == q[np.argmax(q)])
        a = random.choice(a_array[0])
    return a

def run_simulation(q_real, epsilon):
    r_received = np.zeros((time_steps))
    optimal_action_taken = np.zeros((time_steps))

    r_received_n = np.zeros((time_steps))
    optimal_action_taken_n = np.zeros((time_steps))

    q_estimate_n = np.zeros((k))
    n = np.zeros((k))

    q_estimate = np.zeros((k))

    for step in range(time_steps):
        q_real = np.add(q_real, np.random.normal(0, 0.01, (k)))
        optimal_a = np.argmax(q_real)

        action = epsilon_greedy(q_estimate, epsilon)
        r = np.random.normal(q_real[action], sigma)
        q_estimate[action] = q_estimate[action] + alpha * (r - q_estimate[action])
        r_received[step] = r_received[step] + r
        optimal_action_taken[step] = optimal_action_taken[step] + (1 if action == optimal_a else 0)

        action_n = epsilon_greedy(q_estimate_n, epsilon)
        r_n = r if action_n == action else np.random.normal(q_real[action_n], sigma)
        n[action_n] = n[action_n] + 1
        q_estimate_n[action_n] = q_estimate_n[action_n] + ((r_n - q_estimate_n[action_n]) / n[action_n])
        r_received_n[step] = r_received_n[step] + r_n
        optimal_action_taken_n[step] = optimal_action_taken_n[step] + (1 if action_n == optimal_a else 0)

    return r_received, optimal_action_taken, r_received_n, optimal_action_taken_n

r_10 = np.zeros((time_steps))
r_alpha = np.zeros((time_steps))

optimal_action_10 = np.zeros((time_steps))
optimal_action_alpha = np.zeros((time_steps))

for problem in range(num_of_problems):
    bandit_q = np.random.normal(mu, sigma, size=(k))

    r, o, r_n, o_n = run_simulation(bandit_q, 0.1)
    r_10 = np.add(r_10, r_n)
    r_alpha = np.add(r_alpha, r)
    optimal_action_10 = np.add(optimal_action_10, o_n)
    optimal_action_alpha = np.add(optimal_action_alpha, o)
    
plt.plot(r_10 / num_of_problems, color='b')
plt.plot(r_alpha / num_of_problems, color='r')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.show()

plt.plot(optimal_action_10 / num_of_problems, color='b')
plt.plot(optimal_action_alpha / num_of_problems, color='r')
plt.axis([0, time_steps, 0, 1.0])
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.show()