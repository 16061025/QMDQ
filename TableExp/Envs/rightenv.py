import numpy as np
import math

class RightEnv():
    def __init__(self, optimalmean=0.5, optimalvar=1.0, submean=-0.1, subvar=1.0):
        self.submean = submean
        self.subvar = subvar
        self.substd = math.sqrt(self.subvar)
        self.optimalmean = optimalmean
        self.optimalvar = optimalvar
        self.optimalstd = math.sqrt(self.optimalvar)

        self.STATE_S = 0


        self.Right = 0

        self.nState = 1
        self.nAction = 10

        self.state = self.STATE_S
        self.name = "rightenv"

    def reset(self):
        self.state = self.STATE_S
        return self.state

    def step(self, action):
        # A--right
        if action == self.Right:
            reward = np.random.normal(self.optimalmean, self.optimalstd)
        # A--other
        else:
            reward = np.random.normal(self.submean, self.substd)
        self.state = self.STATE_S
        return self.state, reward, False

    def action_number(self, state):
        return self.nAction
