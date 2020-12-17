from environment.EnvironmentManager import *
from environment.CampaignEnvironment import *
from utility.ClairvoyantAlgorithm import *
from learners.AdvertisingLearners import SubCampaignLearner
from environment.BernulliEnvironment import *
from utility.PersonManager import *
from utility.ContextManager import *
from utility.ExperimentManager import *
from utility.Plotter import PlotRegret
import numpy as np


class Task56:
    def __init__(self, budget=10, nArms=10, singlePrice=False):
        self.budget = budget
        self.nArms = nArms
        self.budgets = np.linspace(0.0, self.budget, self.nArms)

        environment = EnvironmentManager()
        self.phaseLabels = environment.phaseLabels
        self.phaseWeights = environment.GetPhaseWeights()
        self.featureLabels = environment.featureLabels
        self.clickFunctions = environment.clickFunctions
        self.sigma = environment.sigma
        self.categories = environment.GetIndexedCategories()
        self.features = environment.features
        self.featureSpace = environment.featureSpace
        self.personType = np.array(environment.probabilities)
        self.possiblePrices = np.array(environment.prices)

        self.singlePrice = singlePrice
        self.optSuperArmReward = self.RunClairvoyant()
        self.optRewardsPerExperiment = []
        self.gptsRewardsPerExperiment = []

    def RunClairvoyant(self):

        # Creo la campagna e le sottocampagne e calcolo i valori delle click function
        env = Campaign(self.budgets, phases=self.phaseLabels, weights=self.phaseWeights)
        for featureLabel in self.featureLabels:
            env.AppendSubCampaign(label=featureLabel, functions=self.clickFunctions[featureLabel])
        clickValues = env.RoundAll()

        # Calcola l'expected value per ogni tipologia di persona per ogni possibile prezzo
        expectedValues = [
            [a * b for a, b in zip(self.possiblePrices, self.personType[i])]
            for i in range(len(self.personType))
        ]

        # Se si utilizza la clairvoyant normale viene preso il masismo di ogni expected value e vi moltiplica
        # per il numero di volte che quella pubblicità viene cliccata
        realExpectedValues = [max(expectedValues[i]) for i in range(len(expectedValues))]
        realValues = [[realExpectedValues[a] * clickValues[a][b]
                       for b in range(self.nArms)]
                      for a in range(len(realExpectedValues))]

        # Calcolo quale partizione di budged tra le varie campagne restituisce la reward maggiore
        optSuperArm = ClairvoyantKnapsack(realValues)

        optSuperArmReward = 0
        # Calcolo la reward per i valori di budged designati
        for (subCampaignID, pulledArm) in enumerate(optSuperArm):
            reward = env.subCampaigns[subCampaignID].Round(pulledArm) * realExpectedValues[subCampaignID]
            optSuperArmReward += reward
        return optSuperArmReward

    def Run(self, nExp=10, horizon=1000):

        # Il valore ottimale è costante per tutto l'intervallo interessato
        self.gptsRewardsPerExperiment = []
        self.optRewardsPerExperiment = [self.optSuperArmReward] * horizon
        for e in range(0, nExp):
            print("E" + str(e) + ": ", end='')
            # Creo l'experiment manager
            environment = BernulliEnvironment(self.possiblePrices, self.personType)
            personManager = PersonManager(self.categories, self.personType, self.features)
            contextManager = ContextManager(len(self.possiblePrices), self.featureSpace,
                                            self.categories, -1, contextKnown=True)
            # Creo io un context per ogni categoria di persona, ogni context ne gestisce una tipologia
            for i in range(len(self.categories)):
                contextManager.AddContext(self.categories[i])
                contextManager.AddContext(self.categories[i])
                contextManager.AddContext(self.categories[i])
            experimentManager = ExperimentManager(personManager, contextManager, environment)

            # Carico la campaign e le sottocampagne (Stesso codice Task1)
            campaignEnvironment = Campaign(self.budgets, phases=self.phaseLabels, weights=self.phaseWeights,
                                           sigma=self.sigma)
            subCampaignLearners = []
            for subCampaignID, featureLabel in enumerate(self.featureLabels):
                campaignEnvironment.AppendSubCampaign(label=featureLabel, functions=self.clickFunctions[featureLabel])
                learner = SubCampaignLearner(arms=self.budgets, label=featureLabel)
                clicks = campaignEnvironment.subCampaigns[subCampaignID].RoundAll()
                samples = [self.budgets, clicks]
                learner.Hyper(samples)
                subCampaignLearners.append(learner)

            rewards = []
            for t in range(0, horizon):

                # Calcolo quanti click sono previsti per ogni sottocampagna per ogni fascia di prezzo
                clickEstimations = []
                for subCampaignLearner in subCampaignLearners:
                    estimate = subCampaignLearner.PullArms()
                    estimate[0] = 0
                    clickEstimations.append(estimate)

                # All'inizio le probabilità sono tutte al 50% quindi si hanno tutte reward pari alla metà del prezzo
                expectedValues = experimentManager.expectedValues

                # Con la clairvoyant normale calcolo il prezzo che mi da il risultato migliore
                if not self.singlePrice:

                    bestExpValues = [max(expectedValues[i]) for i in range(len(expectedValues))]
                    # Calcolo la reward del caso migliore
                    values = [[bestExpValues[a] * clickEstimations[a][b]
                               for b in range(self.nArms)]
                              for a in range(len(expectedValues))]

                    superArm = ClairvoyantKnapsack(values)
                else:

                    # Separo gli expected value in base alle fasce di prezzo per le tre tipologie di persone
                    expectedValues = np.array(expectedValues)
                    expectedValueForBudget = [expectedValues[:, i] for i in range(len(expectedValues[0]))]

                    # Propongo lo stesso prezzo a tutte le tipologie di persona
                    values = []
                    for i in range(len(expectedValueForBudget)):
                        values.append(expectedValueForBudget[i].reshape(3, 1) * clickEstimations)

                    # Calcolo la distribuzione di budget tra le 3 sottocampagne per ogni fascia di prezzo
                    possibleKnapsackAssignments = []
                    optimalKnapsackValues = []
                    for temp in values:
                        # Calcolo la distribuzione di budget
                        knapsackAssignments = ClairvoyantKnapsack(temp)

                        # E la reward per quella fascia di prezzo
                        knapsackValues = []
                        for row in range(len(knapsackAssignments)):
                            knapsackValues.append(temp[row][knapsackAssignments[row]])

                        # Salvo entrambi i risultati
                        possibleKnapsackAssignments.append(knapsackAssignments)
                        optimalKnapsackValues.append(knapsackValues)

                    # Individuo la distribuzione di budget che massimizza la reward
                    optimalPrice = np.argmax(np.sum(optimalKnapsackValues, axis=1))
                    superArm = possibleKnapsackAssignments[optimalPrice]

                # Calcolo il numero di click che avengono in ogni sottocampagna utilizzando il budget ottimale
                clicks = []
                for (subCampaignID, pulledArm) in enumerate(superArm):
                    # Calcolo il numero di click
                    armReward = campaignEnvironment.subCampaigns[subCampaignID].Round(pulledArm)
                    # Aggiorno il gps learner
                    subCampaignLearners[subCampaignID].Update(pulledArm, armReward)
                    # Salvo il numero di click
                    clicks.append(armReward)

                # L'experiment manager farà una bernulli per ogni click per vedere quanti effetti acquisti ci saranno
                # E aggiornerà le expected value di conseguenza
                experimentManager.RunPricingExperiment(clicks)

                if not self.singlePrice:

                    expectedValues = experimentManager.expectedValues

                    # Prendo l'expected value migliore e lo moltiplico per il numero di click previsti
                    bestExpValues = [max(expectedValues[i]) for i in range(len(expectedValues))]
                    superArmReward = [(c * e) for c, e in zip(clicks, bestExpValues)]

                    if t == horizon - 1:
                        bestPricesTP = [np.argmax(expectedValues[i]) for i in range(len(expectedValues))]
                        toPrint = np.array(bestExpValues)
                        print(
                            "Price 1: " + str(self.possiblePrices[bestPricesTP[0]]) + "; Reward" + ": " + "%.2f" %
                            toPrint[0] + " Total reward: " + "%.2f" % superArmReward[0] +
                            " Price 2: " + str(self.possiblePrices[bestPricesTP[1]]) + "; Reward" + ": " + "%.2f" %
                            toPrint[1] + " Total reward: " + "%.2f" % superArmReward[1] +
                            " Price 3: " + str(self.possiblePrices[bestPricesTP[2]]) + "; Reward" + ": " + "%.2f" %
                            toPrint[2] + " Total reward: " + "%.2f" % superArmReward[2]
                        )



                else:

                    # Carico le expected value e le suddivido per fasce di prezzo
                    expectedValues = np.array(experimentManager.expectedValues)
                    expectedValuesPerBudget = [expectedValues[:, i] for i in range(len(expectedValues[0]))]

                    # Recupero il prezzo ottimale
                    optimalEVOptimalPrices = expectedValuesPerBudget[optimalPrice]

                    # Calcolo la reward di questo prezzo
                    superArmReward = [(c * e) for c, e in zip(clicks, optimalEVOptimalPrices)]

                    if t == horizon - 1:
                        toPrint = np.array(optimalEVOptimalPrices)
                        print("Price: " + str(self.possiblePrices[optimalPrice]) + "; Rewards" +
                              ": " + "%.2f" % toPrint[0] + " Total reward: " + "%.2f" % superArmReward[0] +
                              ", " + "%.2f" % toPrint[1] + " Total reward: " + "%.2f" % superArmReward[1] +
                              ", " + "%.2f" % toPrint[2] + " Total reward: " + "%.2f" % superArmReward[2]
                              )

                # Salvo la reward
                rewards.append(sum(superArmReward))

            self.gptsRewardsPerExperiment.append(rewards)
        PlotRegret(self.optRewardsPerExperiment, self.gptsRewardsPerExperiment, tasks=5)
