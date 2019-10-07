import numpy as np
import matplotlib.pyplot as plt
import random


k = 10 # num of bandits
mu = 0 # q*(a) average
sigma = 1 # q*(a) standard deviation

num_of_problems = 2000
time_steps = 1000


def epsilon_greedy(q, epsilon):
    if (random.random() < epsilon):
        a = random.randint(0, len(q) - 1)
    else:
        a_array = np.nonzero(q == q[np.argmax(q)])
        a = random.choice(a_array[0])
    return a

def upper_confidence_bound(q, c, t, n):
    # if there are maximizing actions, use them
    a_array = np.nonzero(n == 0)

    # if there are no maximizing actions, calculate UCB
    if len(a_array[0]) == 0:
        a_array = np.nonzero(q == q[np.argmax(q + c * np.sqrt(np.log(t) / n))])

    # break ties with random choice
    a = random.choice(a_array[0])

    # return the selected action
    return a

def run_simulation(q_real, epsilon, c_degree):
    r_received = np.zeros((time_steps))
    optimal_action_taken = np.zeros((time_steps))

    optimal_a = np.argmax(q_real)

    q_estimate = np.zeros((k))
    n = np.zeros((k))

    for step in range(time_steps):
        if epsilon == 0:
            action = upper_confidence_bound(q_estimate, c_degree, step+1, n)
        else:
            action = epsilon_greedy(q_estimate, epsilon)

        r = np.random.normal(q_real[action], sigma)

        n[action] = n[action] + 1
        q_estimate[action] = q_estimate[action] + ((r - q_estimate[action]) / n[action])
        
        r_received[step] = r_received[step] + r
        optimal_action_taken[step] = optimal_action_taken[step] + (1 if action == optimal_a else 0)

    return r_received, optimal_action_taken

r_10 = np.zeros((time_steps))
r_ucb = np.zeros((time_steps))

optimal_action_10 = np.zeros((time_steps))
optimal_action_ucb = np.zeros((time_steps))

for problem in range(num_of_problems):
    bandit_q = np.random.normal(mu, sigma, size=(k))
    
    r, o = run_simulation(bandit_q, 0.1, 2)
    r_10 = np.add(r_10, r)
    optimal_action_10 = np.add(optimal_action_10, o)
    
    r, o = run_simulation(bandit_q, 0, 2)
    r_ucb = np.add(r_ucb, r)
    optimal_action_ucb = np.add(optimal_action_ucb, o)

plt.plot(r_10 / num_of_problems, color='gray')
plt.plot(r_ucb / num_of_problems, color='b')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.show()

plt.plot(optimal_action_10 / num_of_problems, color='gray')
plt.plot(optimal_action_ucb / num_of_problems, color='b')
plt.axis([0, time_steps, 0, 1.0])
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.show()