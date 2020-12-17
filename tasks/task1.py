from environment.CampaignEnvironment import *
from environment.EnvironmentManager import *
from utility.ClairvoyantAlgorithm import *
from learners.AdvertisingLearners import SubCampaignLearner
from utility.Plotter import PlotRegret
import numpy as np


class Task1:
    def __init__(self, budget=5.0, nArms=6):
        self.budget = budget
        self.nArms = nArms
        self.budgets = np.linspace(0.0, self.budget, self.nArms)

        advertisingEnvironment = EnvironmentManager()
        self.realValues = None
        self.phaseLabels = advertisingEnvironment.phaseLabels
        self.phaseWeights = advertisingEnvironment.GetPhaseWeights()
        self.featureLabels = advertisingEnvironment.featureLabels
        self.clickFunctions = advertisingEnvironment.clickFunctions
        self.sigma = advertisingEnvironment.sigma
        self.OptSuperArmReward = None
        self.optRewardsPerExperiment = []
        self.gptsRewardsPerExperiment = []

    def RunClairvoyant(self):

        # Calcolo i valori delle click function
        optEnv = Campaign(self.budgets, phases=self.phaseLabels, weights=self.phaseWeights)
        for feature_label in self.featureLabels:
            optEnv.AppendSubCampaign(label=feature_label, functions=self.clickFunctions[feature_label])
        realValues = optEnv.RoundAll()
        self.realValues = realValues

        # Calcolo la suddivisione di budget che mi da la miglior reward
        optSuperArm = ClairvoyantKnapsack(realValues)
        optSuperArmReward = 0

        # Calcolo il valore della distribuzione ottimale di budget
        for (subCampaignID, pulledArm) in enumerate(optSuperArm):
            reward = optEnv.subCampaigns[subCampaignID].Round(pulledArm)
            optSuperArmReward += reward
        self.OptSuperArmReward = optSuperArmReward
        return GetDF(realValues, optSuperArm, self.budgets)

    def Run(self, nExp=10, horizon=1000):
        self.gptsRewardsPerExperiment = []

        # Ovviamente la reward ottimale sarà la stessa in ogni round
        self.optRewardsPerExperiment = [self.OptSuperArmReward] * horizon

        # Ripeto lo stesso algoritmo più volte
        for e in range(0, nExp):
            print(str(e),end='')
            # L'environment è lo stesso di quello ottimale
            env = Campaign(self.budgets, phases=self.phaseLabels, weights=self.phaseWeights, sigma=self.sigma)
            subCampaignLearners = []

            # Creo un Learner per ogni SubCampaign
            for subCampaignID, featureLabel in enumerate(self.featureLabels):
                env.AppendSubCampaign(label=featureLabel, functions=self.clickFunctions[featureLabel])
                learner = SubCampaignLearner(arms=self.budgets, label=featureLabel)

                # Faccio il roundAll ma non faccio l'update dei valori, mi serve solo per avere dei valori casuali
                clicks = env.subCampaigns[subCampaignID].RoundAll()
                samples = [self.budgets, clicks]

                # Utilizzo i valori appena calcolati per inizializzare il gpts
                # In questo modo gli hyperparameter non sono troppo sballati
                learner.Hyper(samples)
                subCampaignLearners.append(learner)

            # Ora calcolo i valori reali del gpts
            rewards = []
            for t in range(0, horizon):
                estimations = []
                for subCampaignLearner in subCampaignLearners:
                    estimate = subCampaignLearner.PullArms()
                    estimate[0] = 0
                    estimations.append(estimate)
                # Una volta estratti i valori dai vari arm calcolo quelli migliori
                superArm = ClairvoyantKnapsack(estimations)

                superArmReward = 0
                for (subCampaignID, pulled_arm) in enumerate(superArm):
                    # Calcolo il valore della subcampaign per tutte e tre le fasi
                    arm_reward = env.subCampaigns[subCampaignID].Round(pulled_arm)
                    superArmReward += arm_reward

                    # Aggiorno il learner
                    subCampaignLearners[subCampaignID].Update(pulled_arm, arm_reward)
                rewards.append(superArmReward)
            self.gptsRewardsPerExperiment.append(rewards)
        print(' ')
        PlotRegret(self.optRewardsPerExperiment, self.gptsRewardsPerExperiment, tasks=1)
