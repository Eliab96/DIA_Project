from environment.BernulliEnvironment import *
from environment.EnvironmentManager import *
from utility.PersonManager import *
from utility.ContextManager import *
from utility.ExperimentManager import *
from utility.Plotter import PlotCumulativeRegret


class Task34:
    def __init__(self):
        environment = EnvironmentManager()

        # Categories contiene la scelta delle feature (teen/notteen) (employed/notemployed)
        self.categories = environment.GetIndexedCategories()
        self.features = environment.features
        self.featureSpace = environment.featureSpace
        self.probabilities = np.array(environment.probabilities)
        self.possiblePrices = np.array(environment.prices)
        self.nArms = len(self.possiblePrices)
        self.expLogs = []
        self.betaParams = []
        self.week = -1

    def RunClairvoyant(self):
        return np.max(np.multiply(self.probabilities, self.possiblePrices), axis=1).T

    def Run(self, nExp, horizon, week=-1):
        self.expLogs = []
        self.week = week
        for e in range(nExp):
            print("E" + str(e) + ": ")
            # L'esperimento viene completamente gestito dal context generator
            environment = BernulliEnvironment(self.possiblePrices, self.probabilities)
            personManager = PersonManager(self.categories, self.probabilities, self.features)
            contextManager = ContextManager(self.nArms, self.featureSpace, self.categories, week)
            experimentManager = ExperimentManager(personManager, contextManager, environment)

            self.expLogs.append(experimentManager.PlayExperiment(horizon))
            self.betaParams.append(experimentManager.contextManager.contextsSet[0].learner.betaParameters)

        PlotCumulativeRegret(self.expLogs, self.probabilities, self.possiblePrices, self.RunClairvoyant())
