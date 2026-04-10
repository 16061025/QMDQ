import numpy as np
import math

class SameActionEnv():
    def __init__(self, mean=0, var=1.0):
        self.mean = mean
        self.var = var
        self.state = 0


        self.nState = 1
        self.nAction = 10
        self.STATE_S = 0


    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        reward = np.random.normal(self.mean, self.var)
        return self.state, reward, False

    def action_number(self, state):
        return self.nAction
