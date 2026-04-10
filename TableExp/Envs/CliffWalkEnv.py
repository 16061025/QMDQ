import numpy as np
class CliffWalkingEnv():
    def __init__(self, rows=4, cols=12, cliff_reward=-100, step_reward=-1, goal_reward=1):
        self.nrow = rows
        self.ncol = cols
        self.START_STATE = 0  # 左下角
        self.GOAL_STATE = self.nrow * self.ncol - 1  # 右下角
        self.cur_x = 0
        self.cur_y = 0

        # 4种动作: 0:上, 1:下, 2:左, 3:右
        self.changes = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.nAction = 4
        self.nState = self.nrow * self.ncol
        self.name = "cliff_walking"
        self.STATE_S = self.START_STATE

        # 奖励设置
        self.cliff_reward = cliff_reward
        self.step_reward = step_reward
        self.goal_reward = goal_reward

        # 悬崖区域：第0行的第1到第ncol-2列
        self.cliff_states = []
        for x in range(1, self.ncol - 1):
            self.cliff_states.append(0 * self.ncol + x)

    def reset(self):
        self.state = self.START_STATE
        self.cur_x = 0
        self.cur_y = 0
        return self.state

    def step(self, action):
        next_x = min(self.ncol - 1, max(0, self.cur_x + self.changes[action][0]))
        next_y = min(self.nrow - 1, max(0, self.cur_y + self.changes[action][1]))
        next_state = next_y * self.ncol + next_x

        # 检查是否到达目标
        if next_state == self.GOAL_STATE:
            reward = self.goal_reward
            done = True
        # 检查是否跌入悬崖
        elif next_state in self.cliff_states:
            reward = self.cliff_reward
            done = True
        else:
            # 普通移动
            reward = self.step_reward
            done = False

        # 更新状态
        self.state = next_state
        self.cur_x = next_x
        self.cur_y = next_y

        return self.state, reward, done

    def action_number(self, state):
        return self.nAction