import numpy as np

'''
    Classe che gestisce la campagna pubblicitaria e le sue sottocampagne.
    Utilizzo lo stesso metodo visto durante il laboratorio ripetuto 1 volta per ogni sottocampagna.
'''


class Campaign:
    def __init__(self, budgets, phases, weights, sigma=0.0):
        self.subCampaigns = []
        self.phases = phases
        self.weights = weights
        self.budgets = budgets
        self.sigma = sigma

    def AppendSubCampaign(self, label, functions):
        self.subCampaigns.append(
            SubCampaign(label, self.budgets, functions, self.sigma, self.weights)
        )

    def Round(self, subCampaignId, pulledArm, phase=None):
        return self.subCampaigns[subCampaignId].Round(pulledArm, phase)

    def RoundAll(self, phase=None):
        table = []
        for subCampaign in self.subCampaigns:
            table.append(subCampaign.RoundAll(phase))
        return table


class SubCampaign:
    def __init__(self, label, budgets, functions, sigma, weights):
        self.label = label
        self.weights = weights
        self.budgets = budgets
        self.nPhases = len(functions)
        self.phases = [SubPhase(budgets, functions[i], sigma) for i in range(self.nPhases)]

    def Round(self, pulledArm, phase=None):
        if phase is None:
            reward = 0

            # La reward viene presa da una distribuzione normale e viene moltiplicata per il peso di ogni fase
            # Ovviamente la sommatoria dei pesi è 1
            for i in range(self.nPhases):
                reward += self.weights[i] * self.phases[i].Round(pulledArm)
            return reward
        else:
            return self.phases[phase].Round(pulledArm)

    def RoundAll(self, phase=None):
        rewards = []
        for i in range(len(self.budgets)):
            rewards.append(self.Round(i, phase))
        return rewards


'''
    Questa classe si occupa semplicemente ti tenere salvate la media e la sigma per ogni sottofase
'''


class SubPhase:
    def __init__(self, budgets, function, sigma):
        self.means = function(budgets)
        self.sigmas = np.ones(len(budgets)) * sigma

    def Round(self, pulledArm):
        return np.random.normal(self.means[pulledArm], self.sigmas[pulledArm])
