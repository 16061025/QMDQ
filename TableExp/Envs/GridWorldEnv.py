import numpy as np
class GridWorldEnv():
    def __init__(self, row=3, col=3, gridr1=-6, gridr2=4, gridgr1=-30, gridgr2=40):
        self.nrow = row
        self.ncol = col
        self.STATE_S = 0
        self.STATE_G = self.nrow * self.ncol - 1
        self.cur_x = 0
        self.cur_y = 0
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        self.changes = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.nAction = 4
        self.nState = self.nrow * self.ncol
        self.name = "grid"

        self.gridr1 = gridr1
        self.gridr2 = gridr2
        self.gridgr1 = gridgr1
        self.gridgr2 = gridgr2


    def reset(self):
        self.state = self.STATE_S
        self.cur_x = 0
        self.cur_y = 0
        return self.state

    def step(self, action):

        next_x = min(self.ncol - 1, max(0, self.cur_x + self.changes[action][0]))
        next_y = min(self.nrow - 1, max(0, self.cur_y + self.changes[action][1]))
        next_state = next_y * self.ncol + next_x

        if self.state != self.STATE_G:
            reward = np.random.choice([self.gridr1, self.gridr2])
            done = False
        elif self.state == self.STATE_G:
            reward = np.random.choice([self.gridgr1, self.gridgr2])
            done = False
            next_state = self.STATE_S
            next_x = 0
            next_y = 0
        self.state = next_state
        self.cur_x = next_x
        self.cur_y = next_y
        return self.state, reward, done


    def action_number(self, state):
        return self.nAction
