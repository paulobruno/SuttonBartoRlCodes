import numpy as np

from gym.utils import seeding
from gym import error, spaces

class RandomWalk:
    def __init__(self, states=5):
        self.__version__ = "0.0.1"
        self.states = states

        self._action_set = [0, 1]
        self.action_space = spaces.Discrete(len(self._action_set))

        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # define next state table
        self.nextState = np.zeros((2, self.states), dtype=int) # two actions, n states
        self.nextState[0] = [(i-1) for i in range(self.states)]
        self.nextState[1] = [(i+1) for i in range(self.states)]
        self.nextState[0,0] = -1
        self.nextState[1,self.states-1] = -1

        # define reward table
        self.reward = np.zeros((2, self.states), dtype=int)
        self.reward[1, self.states-1] = 1

        # start is always the center state
        self.current_state = int(self.states / 2)

        return self.current_state
        
    def step(self, action):
        if (action != 0) and (action != 1):
            raise error.Error('Invalid action: ' + str(action))

        old_state = self.current_state
        self.current_state = self.nextState[action, old_state]
        r = self.reward[action, old_state]
        done = True if self.current_state == -1 else False # -1 is the terminal state

        return self.current_state, r, done, {}
        
    # TODO: melhorar essa maneira de desenhar o grid, ver exemplo taxi
    def render(self):
        for i in range(self.states):
            if i == self.current_state:
                print(">", end="")
            print(str(i), end="")
            if i == self.current_state:
                print("<", end="")
            print(" ", end="")
        print()

    def close(self):
        return
        
    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

ACTION_MEANING = {
    0: "LEFT",
    1: "RIGHT"
}


# TD(0) agent
num_episodes = 10000
alpha = 0.1
gamma = 0.9

num_states = 5
v_s = np.full((num_states+2), 0.5, dtype=float)
v_s[0] = 0
v_s[-1] = 0

rmse = np.zeros((num_runs), dtype=float)

env = RandomWalk(num_states)

for _ in range(num_episodes):
    old_state = env.reset()
    done = False

    while not done:
        #env.render()
        #print(v_s[1:-1])
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        v_s[old_state] = v_s[old_state] + alpha * (reward + gamma * v_s[next_state] - v_s[old_state])

        old_state = next_state

    #print('r: ' + str(reward))
    #print(v_s[1:-1])
print(v_s[1:-1])
env.close()