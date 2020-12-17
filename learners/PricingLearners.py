import numpy as np
import math

'''
    Thompson Sampling Learner
'''


class TSLearner:
    def __init__(self, nArms):
        self.nArms = nArms
        self.t = 0
        self.rewardsPerArm = [[] for i in range(nArms)]
        self.collectedRewards = np.array([])
        self.betaParameters = np.ones((nArms, 2))
        self.meanArm = {i: 0 for i in range(nArms)}

    def PullArm(self, values):
        betaSamples = np.random.beta(self.betaParameters[:, 0], self.betaParameters[:, 1])
        return np.argmax(np.multiply(betaSamples, values))

    def Update(self, pulledArm, reward):

        # Aggiorno i betaparameter in base alla soluzione della Bernulli
        self.t += 1
        if reward > 0:
            reward = 1
        elif reward <= 0:
            reward = 0
        self.rewardsPerArm[pulledArm].append(reward)
        self.collectedRewards = np.append(self.collectedRewards, reward)
        self.betaParameters[pulledArm, 0] = self.betaParameters[pulledArm, 0] + reward
        self.betaParameters[pulledArm, 1] = self.betaParameters[pulledArm, 1] + 1.0 - reward

    def ArmProbability(self, arm):

        # Calcolo la probabilità di successo
        alpha = self.betaParameters[arm][0]
        beta = self.betaParameters[arm][1]
        return alpha / (alpha + beta)

    def ExpectedValue(self, arm, price):
        return self.ArmProbability(arm) * price

    # Calcolo il prezzo ottimale
    def OptimalArm(self, values):
        opt = 0
        optValue = self.ExpectedValue(0, values[0])
        for i in range(self.nArms):
            temp = self.ExpectedValue(i, values[i])
            if temp > optValue:
                opt = i
                optValue = temp
        return opt

    # Expected value del prezzo ottimale
    def OptimalExpectedValue(self, values):
        optArm = self.OptimalArm(values)
        return self.ExpectedValue(optArm, values[optArm])
