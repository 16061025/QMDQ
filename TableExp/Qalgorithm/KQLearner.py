import numpy as np
import math

class KQLearner():
    def __init__(self, K=2, epsilon=1.0, adpepsilon=True, gamma=0.95, learningRate=1.0, adplearningRate=True, lrexp=0.8, epsilonexp=0.5, env=None):
        self.learningRate = learningRate
        self.lrexp = lrexp
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon

        self.gamma = gamma
        self.K = K

        self.env = env
        self.init_Q_tables()

        self.updateQinex = 0


    def init_Q_tables(self):
        #np.random.seed(42)
        self.Q_tables = np.random.normal(0, 0.01, size=(self.K, self.env.nState, self.env.nAction))
        self.Count_S_A = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S = np.zeros(shape=self.env.nState)

    def explore(self, state):
        self.Count_S[state] += 1

        epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)

        action_number = self.env.action_number(state)
        if np.random.random() >= epsilon_temp:

            selector_indices = np.arange(self.K)
            selector_Q_tables = self.Q_tables[selector_indices, state, :action_number]
            selector_Q_bar = np.mean(selector_Q_tables, axis=0)
            action = np.argmax(selector_Q_bar)
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):



        action_number = self.env.action_number(next_state)
        self.Count_S_A[state, action] += 1

        lr = self.learningRate / np.power(self.Count_S_A[state, action], self.lrexp)

        for i in range(self.K):
            td_target = reward
            if not done:
                td_target += self.gamma * self.Q_tables[i][next_state][np.argmax(self.Q_tables[i][next_state][:action_number])]
            self.Q_tables[i][state, action] += lr * (td_target - self.Q_tables[i][state, action])


    def maxQ(self, state):
        action_number = self.env.action_number(state)
        Q_s_bar = np.mean(self.Q_tables[:, state, :action_number], axis=0)
        return max(Q_s_bar)
