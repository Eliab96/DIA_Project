from environment.EnvironmentManager import *
from environment.CampaignEnvironment import *
from utility.ClairvoyantAlgorithm import *
from learners.AdvertisingLearners import SubCampaignLearner, SWSubCampaignLearner
from utility.Plotter import PlotSWRegret
import numpy as np


class Task2:
    def __init__(self, budget=10.0, n_arms=10, sampleFactor=10):
        self.budget = budget
        self.nArms = n_arms
        self.budgets = np.linspace(0.0, self.budget, self.nArms)
        env = EnvironmentManager()
        self.phaseLabels = env.phaseLabels
        self.phaseWeights = env.GetPhaseWeights()
        self.phaseList = env.GetPhaseList(sampleFactor)
        self.phaseLen = len(self.phaseList)
        self.featureLabels = env.featureLabels
        self.clickFunctions = env.clickFunctions
        self.sigma = env.sigma
        self.optimalSuperArmRewards = self.RunClairvoyant()
        self.optRewardsPerExperiment = []
        self.gptsRewardsPerExperiment = []
        self.swgptsReardsPerExperiment = []
        self.windowSize = None

    def RunClairvoyant(self):
        optEnv = Campaign(self.budgets, phases=self.phaseLabels, weights=self.phaseWeights)
        for featureLabel in self.featureLabels:
            optEnv.AppendSubCampaign(label=featureLabel, functions=self.clickFunctions[featureLabel])
        optimalSuperArmRewards = []

        # Calcolo il valore di Clairvoyant, a differenza della fase 1 non unisco tutte le fasi
        for phase in range(len(self.phaseLabels)):

            # Calcolo i valori solo per questa fase
            realValues = optEnv.RoundAll(phase=phase)
            optimalSuperArmPhase = ClairvoyantKnapsack(realValues)
            optimalSuperArmReward = 0

            # Calcolo la reward per il superArm ottimale
            for (subCampaignID, pulledArm) in enumerate(optimalSuperArmPhase):
                optimalSuperArmReward += optEnv.subCampaigns[subCampaignID].Round(pulledArm, phase=phase)
            optimalSuperArmRewards.append(optimalSuperArmReward)
        return optimalSuperArmRewards

    def Run(self, nExp=10, horizon=10, windowSize=10):
        self.optRewardsPerExperiment = []

        # Le optimal reward le ho calcolate precedentemente
        for t in range(0, horizon):
            self.optRewardsPerExperiment.append(self.optimalSuperArmRewards[self.phaseList[t % self.phaseLen]])

        self.swgptsReardsPerExperiment = []
        self.gptsRewardsPerExperiment = []
        for e in range(0, nExp):
            print(str(e), end='')
            # Creo campagna e sottocampagne
            env = Campaign(self.budgets, phases=self.phaseLabels, weights=self.phaseWeights, sigma=self.sigma)
            subCampaignLearner = []
            swSubCampaignLearner = []
            for subCampaignID, featureLabel in enumerate(self.featureLabels):
                env.AppendSubCampaign(label=featureLabel, functions=self.clickFunctions[featureLabel])
                clicks = env.subCampaigns[subCampaignID].RoundAll(phase=0)
                samples = [self.budgets, clicks]

                # Esattamente come nella task1 solo che creo due learner, uno normale e uno SW
                # Normale
                learner = SubCampaignLearner(arms=self.budgets, label=featureLabel)
                learner.Hyper(samples)
                subCampaignLearner.append(learner)

                # SW
                swSCL = SWSubCampaignLearner(arms=self.budgets, label=featureLabel, windowSize=windowSize)
                swSCL.Hyper(samples)
                swSubCampaignLearner.append(swSCL)

            swRewards = []
            rewards = []
            for t in range(0, horizon):

                # Esatttamente come nella task 1 con l'aggiunta della fase
                # Normale
                estimations = []
                for subc_learner in subCampaignLearner:
                    estimate = subc_learner.PullArms()
                    estimate[0] = 0
                    estimations.append(estimate)
                super_arm = ClairvoyantKnapsack(estimations)
                super_arm_reward = 0
                for (subCampaignID, pulled_arm) in enumerate(super_arm):
                    arm_reward = env.subCampaigns[subCampaignID].Round(
                        pulled_arm, phase=self.phaseList[t % self.phaseLen])
                    super_arm_reward += arm_reward
                    subCampaignLearner[subCampaignID].Update(pulled_arm, arm_reward)
                rewards.append(super_arm_reward)

                # SW
                estimations = []
                for SW_s_learner in swSubCampaignLearner:
                    estimate = SW_s_learner.PullArms()
                    estimate[0] = 0
                    estimations.append(estimate)
                super_arm = ClairvoyantKnapsack(estimations)
                super_arm_reward = 0
                for (subCampaignID, pulled_arm) in enumerate(super_arm):
                    arm_reward = env.subCampaigns[subCampaignID].Round(
                        pulled_arm, phase=self.phaseList[t % self.phaseLen])
                    super_arm_reward += arm_reward
                    swSubCampaignLearner[subCampaignID].Update(pulled_arm, arm_reward, t)
                swRewards.append(super_arm_reward)

            self.swgptsReardsPerExperiment.append(swRewards)
            self.gptsRewardsPerExperiment.append(rewards)
        print(' ')
        PlotSWRegret(self.optRewardsPerExperiment,
                     self.gptsRewardsPerExperiment,
                     self.swgptsReardsPerExperiment)
