import numpy as np
import matplotlib.pyplot as plt
import random


k = 10 # num of bandits
mu = 0 # q*(a) average
sigma = 1 # q*(a) standard deviation

num_of_problems = 2000
time_steps = 1000


def run_simulation(q_real, alpha, baseline):
    r_received = np.zeros((time_steps))
    optimal_action_taken = np.zeros((time_steps))

    optimal_a = np.argmax(q_real)

    preference = np.zeros((k))
    pi = np.full((k), 1.0 / k, dtype=float)

    for step in range(time_steps):
        action = np.random.choice((k), p=pi)

        r = np.random.normal(q_real[action], sigma)

        r_baseline = np.sum(r_received) / step if baseline and step != 0 else 0

        p_action = preference[action]
        preference = preference - alpha * (r - r_baseline) * pi
        preference[action] = p_action + alpha * (r - r_baseline) * (1 - pi[action])
        pi = np.exp(preference) / sum(np.exp(preference))
                
        r_received[step] = r_received[step] + r
        optimal_action_taken[step] = optimal_action_taken[step] + (1 if action == optimal_a else 0)

    return r_received, optimal_action_taken

r_10_0 = np.zeros((time_steps))
r_10_4 = np.zeros((time_steps))
r_40_0 = np.zeros((time_steps))
r_40_4 = np.zeros((time_steps))

optimal_action_10_0 = np.zeros((time_steps))
optimal_action_10_4 = np.zeros((time_steps))
optimal_action_40_0 = np.zeros((time_steps))
optimal_action_40_4 = np.zeros((time_steps))

for problem in range(num_of_problems):
    bandit_q = np.random.normal(4, sigma, size=(k))

    r, o = run_simulation(bandit_q, 0.1, False)
    r_10_0 = np.add(r_10_0, r)
    optimal_action_10_0 = np.add(optimal_action_10_0, o)
    
    r, o = run_simulation(bandit_q, 0.1, True)
    r_10_4 = np.add(r_10_4, r)
    optimal_action_10_4 = np.add(optimal_action_10_4, o)
    
    r, o = run_simulation(bandit_q, 0.4, False)
    r_40_0 = np.add(r_40_0, r)
    optimal_action_40_0 = np.add(optimal_action_40_0, o)
    
    r, o = run_simulation(bandit_q, 0.4, True)
    r_40_4 = np.add(r_40_4, r)
    optimal_action_40_4 = np.add(optimal_action_40_4, o)

plt.plot(r_10_0 / num_of_problems, color='#604800')
plt.plot(r_40_0 / num_of_problems, color='#c3b17d')
plt.plot(r_40_4 / num_of_problems, color='#8fdaf8')
plt.plot(r_10_4 / num_of_problems, color='b')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.show()

plt.plot(optimal_action_10_0 / num_of_problems, color='#604800')
plt.plot(optimal_action_40_0 / num_of_problems, color='#c3b17d')
plt.plot(optimal_action_40_4 / num_of_problems, color='#8fdaf8')
plt.plot(optimal_action_10_4 / num_of_problems, color='b')
plt.axis([0, time_steps, 0, 1.0])
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.show()