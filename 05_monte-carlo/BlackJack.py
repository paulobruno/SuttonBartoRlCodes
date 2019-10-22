import numpy as np

# cards distribution in a suit
a = np.array([i if i < 10 else 10 for i in range(1,14)])

# initialize empty player cards

# draw intial cards
for p in range(2):
    for _ in range(2):
        x = np.random.randint(0,13)
        players[p].append(a[x])

print(players)

ACTION_MEANINGS: {
    0: "HIT",
    1: "STICK",
}


import numpy as np

from gym.utils import seeding
from gym import error, spaces


class BlackJack:
    def __init__(self, states=5):
        self.__version__ = "0.0.1"

        self._action_set = [0, 1]
        self.action_space = spaces.Discrete(len(self._action_set))

        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.sum = 0
        self.dealer = 0
        self.ace = 0
        self.state = np.array([self.sum, self.dealer, self.ace])

        return self.state
        
    def step(self, action):
        #case HIT:
        #x = np.random.randint(...)
        #sum += c[x]
        #if sum > 21:
        #    sum -= ace * 10
        #    ace = 0
        #if sum > 21:
        #    perdeu -> r = -1
        #return ...

        if (action != 0) and (action != 1):
            raise error.Error('Invalid action: ' + str(action))

        old_state = self.current_state
        self.current_state = self.nextState[action, old_state]
        r = self.reward[action, old_state]
        done = True if self.current_state == -1 else False # -1 is the terminal state
        
        #if sum_p1 > sum_p2:
        #    p1 venceu
        #if sum_p2 > sum_p1:
        #    p2 venceu
        #else
        #    draw

        return self.current_state, r, done, {}
        
    def render(self):
        for i in range(self.states):
            print(">" + str(i) + "<" if i == self.current_state else str(i), end=" ")
        print()

    def close(self):
        return
        
    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    ACTION_MEANING = {
        0: "HIT",
        1: "STICK"
    }


env = BlackJack()

observation = env.reset()

for _ in range(10):
    env.render()
    action = env.action_space.sample()
    print("action: " + env.get_action_meanings()[action])
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        print("r: " + str(reward))
        print()
    
    #pi(state):
        #if sum > 19 and sum < 22:
        #    a = STICK
        #else
        #    a = HIT
        #obs = env.step(a)
        
env.close()