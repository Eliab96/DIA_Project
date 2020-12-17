import numpy as np


class BernulliEnvironment:
    def __init__(self, possiblePrices, probabilities):
        self.possiblePrices = possiblePrices
        self.probabilities = probabilities
        self.time = 0

    def Round(self, pCategory, pulledArm):
        # Data una probabilità mi restituisce il guadagno
        # Se è fallimento 0 altrimenti il prezzo concordato
        p = self.probabilities[pCategory][pulledArm]
        reward = np.random.binomial(1, p)
        self.time += 1
        return reward * self.possiblePrices[pulledArm]
