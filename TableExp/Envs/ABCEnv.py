import numpy as np

class ABCEnv():
    def __init__(self, meanB=-0.1, meanC=0):
        self.meanB = meanB
        self.meanC = meanC
        self.nState = 3
        self.nAction = 10
        self.stateA = 0
        self.stateB = 1
        self.stateC = 2
        self.state = self.stateA

        self.Left = 0
        self.Right = 1

        self.STATE_S = self.stateA

    def reset(self):
        self.state = self.stateA
        return self.state

    def step(self, action):
        if self.state == self.stateA:
            if action == self.Right:
                reward = 0
                done = False
                self.state = self.stateC
            else:
                reward = 0
                done = False
                self.state = self.stateB

        elif self.state == self.stateB:
            reward = np.random.normal(self.meanB, 1)
            done = True
        elif self.state == self.stateC:
            reward = np.random.normal(self.meanC, 1)
            done = True

        return self.state, reward, done

    def action_number(self, state):
        if state == self.stateA:
            action_number = 2
        elif state == self.stateB or state == self.stateC:
            action_number = 10
        return action_number