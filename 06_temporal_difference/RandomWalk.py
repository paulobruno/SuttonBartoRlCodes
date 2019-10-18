import numpy as np

from gym.utils import seeding
from gym import error, spaces

r_L = 0
r_R = 1

gamma = 0.9

class RandomWalk:
    def __init__(self, width=5, height=5):
        self.__version__ = "0.0.1"
        self.width = width
        self.height = height

        self._action_set = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(len(self._action_set))

        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._values = np.zeros((self.width, self.height))
        self.state = [self.np_random.randint(self.width), 
                      self.np_random.randint(self.height)]
        return np.array(self.state)
        
    def step(self, action):
        done = False
        reward = 0

        if action == 0: # up
            if self.state[0] == 0:
                reward = -1
            elif self.state == [A[0]+1, A[1]]:
                reward = r_A
                self.state = list(A_n)
            elif self.state == [B[0]+1, B[1]]:
                reward = r_B
                self.state = list(B_n)
            else:
                self.state[0] = self.state[0] - 1
        elif action == 1: # down
            if self.state[0] == self.height-1:
                reward = -1
            elif self.state == [A[0]-1, A[1]]:
                reward = r_A
                self.state = list(A_n)
            elif self.state == [B[0]-1, B[1]]:
                reward = r_B
                self.state = list(B_n)
            else:
                self.state[0] = self.state[0] + 1
        elif action == 2: # left
            if self.state[1] == 0:
                reward = -1
            elif self.state == [A[0], A[1]+1]:
                reward = r_A
                self.state = list(A_n)
            elif self.state == [B[0], B[1]+1]:
                reward = r_B
                self.state = list(B_n)
            else:
                self.state[1] = self.state[1] - 1
        elif action == 3: # right
            if self.state[1] == self.width-1:
                reward = -1
            elif self.state == [A[0], A[1]-1]:
                reward = r_A
                self.state = list(A_n)
            elif self.state == [B[0], B[1]-1]:
                reward = r_B
                self.state = list(B_n)
            else:
                self.state[1] = self.state[1] + 1
        else:
            raise error.Error('Invalid action: ' + str(action))
            
        return np.array(self.state), reward, done, {}
        
    # TODO: melhorar essa maneira de desenhar o grid, ver exemplo taxi
    def render(self):
        for i in range(self.height):
            print('+', end='')
            for j in range(self.width):
                print("-------+", end='')
            print('\n', end='|')
            for j in range(self.width):
                print("{:6.2f}".format(self._values[i, j]) + ' ', end='|')
            print()
        print('+', end='')
        for j in range(self.width):
            print("-------+", end='')
        print()

    def close(self):
        return
        
    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def _reward(self, state, action):
        if state == A:
            return r_A
        elif state == B:
            return r_B
        else:
            return 0

ACTION_MEANING = {
    0: "LEFT",
    1: "RIGHT"
}