import numpy as np
#import matplotlib.pyplot as plt

class MytestQLearner:
    def __init__(self, Mixratio=0.2, epsilon=1.0, adpepsilon=True,  gamma=0.95, learningRate=1.0, adplearningRate=True, lrexp=0.8, epsilonexp=0.5, env=None):
        self.UpdateABProbability = Mixratio
        self.UpdateAProbability = (1-self.UpdateABProbability)/2
        self.learningRate = learningRate
        self.lrexp = lrexp
        self.epsilonexp = epsilonexp
        self.epsilon = epsilon
        self.gamma = gamma
        self.adpepsilon = adpepsilon
        self.adplearningRate = adplearningRate
        self.env = env
        self.init_Q_table()
        self.trajectory = []
        self.step_count = 0

    def init_Q_table(self):
        self.Q1 = np.random.normal(0, 0.01, size=(self.env.nState, self.env.nAction))
        self.Q2 = np.random.normal(0, 0.01, size=(self.env.nState, self.env.nAction))
        #self.Q2 = np.copy(self.Q1)
        self.Count_S_A_1 = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S_A_2 = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S_A_single = np.zeros(shape=(self.env.nState, self.env.nAction))
        self.Count_S = np.zeros(shape=self.env.nState)

    def explore(self, state):
        self.Count_S[state] += 1
        if self.adpepsilon:
            epsilon_temp = self.epsilon / np.power(self.Count_S[state], self.epsilonexp)
        else:
            epsilon_temp = self.epsilon
        action_number = self.env.action_number(state)
        if np.random.random() >= epsilon_temp:
            Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
            action = np.argmax(Q3[:])
        else:
            action = np.random.choice(action_number)
        return action

    def learning(self, state, action, reward, next_state, done):
        Y1 = reward
        Y2 = reward
        Y =reward
        prob = np.random.random()
        if prob < self.UpdateAProbability:
            self.Count_S_A_1[state][action] += 1
            if self.adplearningRate:
                lr_1 = self.learningRate / np.power(self.Count_S_A_1[state][action], self.lrexp)
            else:
                lr_1 = self.learningRate
            if not done:
                action_number = self.env.action_number(next_state)
                Y += self.gamma * self.Q2[next_state][np.argmax(self.Q1[next_state][:action_number])]
            self.Q1[state][action] += lr_1 * (Y - self.Q1[state][action])
        elif prob >= 1-self.UpdateABProbability:
            self.Count_S_A_1[state][action] += 1
            self.Count_S_A_2[state][action] += 1
            if self.adplearningRate:
                lr_1 = self.learningRate / np.power(self.Count_S_A_1[state][action], self.lrexp)
                lr_2 = self.learningRate / np.power(self.Count_S_A_2[state][action], self.lrexp)
            else:
                lr_1 = self.learningRate
                lr_2 = self.learningRate
            if not done:
                action_number = self.env.action_number(next_state)
                Y1 += self.gamma * self.Q1[next_state][np.argmax(self.Q1[next_state][:action_number])]
                Y2 += self.gamma * self.Q2[next_state][np.argmax(self.Q2[next_state][:action_number])]
            self.Q1[state][action] += lr_1 * (Y1 - self.Q1[state][action])
            self.Q2[state][action] += lr_2 * (Y2 - self.Q2[state][action])
        else:
            self.Count_S_A_2[state][action] += 1
            if self.adplearningRate:
                lr_2 = self.learningRate / np.power(self.Count_S_A_2[state][action], self.lrexp)
            else:
                lr_2 = self.learningRate
            if not done:
                action_number = self.env.action_number(next_state)
                Y += self.gamma * self.Q1[next_state][np.argmax(self.Q2[next_state][:action_number])]
            self.Q2[state][action] += lr_2 * (Y - self.Q2[state][action])

        if prob < self.UpdateAProbability:
            lr = lr_1
        elif prob >= 1-self.UpdateABProbability:
            lr = lr_2
        else:
            lr = lr_2
        Qpre = 0
        Qafter = 0
        self.Count_S_A_single[state][action] += 1
        self.trajectory.append(
            [state, action, reward, next_state, done, lr, self.Count_S[state], self.Count_S_A_single[state][action], Qpre, Qafter, Y, self.step_count
             ])
        self.step_count += 1



    def maxQ(self, state):
        action_number = self.env.action_number(state)
        Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
        return max(Q3)

    def avgQ(self):
        return  np.stack([self.Q1, self.Q2], axis=0)

    def stateQaction(self, state):
        action_number = self.env.action_number(state)
        Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
        action = np.argmax(Q3[:])
        q = np.max(Q3)
        return q, action

    def getQSA(self, state, action):
        action_number = self.env.action_number(state)
        Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
        return Q3[action]


    def loadQ_tables(self, Q_tables):
        self.Q1 = Q_tables[0]
        self.Q2 = Q_tables[1]

    def getcountSA(self):
        #return np.stack([self.Count_S_A_1, self.Count_S_A_2], axis=0)
        return self.Count_S_A_single
