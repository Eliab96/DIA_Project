from learners.PricingLearners import TSLearner
import numpy as np


def ComputeComplemetaryFeature(feature):
    features_set = (("e", "n"), ("t", "a"))
    for variable in features_set:
        if feature in variable:
            for f in variable:
                if f != feature:
                    return f


class Context:
    def __init__(self, contextID, subspace, learner, logRewards=[]):
        self.contextID = contextID
        self.subspace = subspace
        self.learner = learner
        self.numVar = 2
        self.logRewards = logRewards

    def Update(self, features_person, pulled_arm, reward):
        self.learner.Update(pulled_arm, reward)
        self.logRewards.append((features_person, pulled_arm, reward))

    def SubLogFeature(self, feature):
        subLog = []
        for i in range(len(self.logRewards)):
            if feature in self.logRewards[i][0]:
                subLog.append(self.logRewards[i])
        return subLog

    def LearnerSubContext(self, log):
        # Il learner fa il training su tutto il sublog
        learner = TSLearner(self.learner.nArms)
        for i in range(len(log)):
            learner.Update(log[i][1], log[i][2])
        return learner

    def Split(self, possiblePrices):
        possibleSplits = []
        # Conto quante diverse feature contiene questo context
        countFeatureVal = [[] for x in range(self.numVar)]
        for t in self.subspace:
            for var in range(self.numVar):
                if t[var] not in countFeatureVal[var]:
                    countFeatureVal[var].append(t[var])
        # Se la feature contiene più di due valiri allora è possibile farci lo split
        splittableFeatures = [1 if len(x) >= 2 else 0 for x in countFeatureVal]
        for index, var in enumerate(splittableFeatures):
            if var == 0:
                pass
            else:
                # La split contidition prevede che la media dei due prezzi dei sub context sia maggiore di quello iniziale
                splitCondition = []
                valAfterSplit = self.ValAfterSplit(countFeatureVal[index][0], possiblePrices)
                if valAfterSplit[0] > self.learner.OptimalExpectedValue(possiblePrices):
                    splitCondition = [countFeatureVal[index][0], valAfterSplit[0], valAfterSplit[1], valAfterSplit[2]]
                # Inserisco un'altra possibile split
                if len(splitCondition) > 0:
                    possibleSplits.append(splitCondition)
        # Restituisco solo quella migliore
        if len(possibleSplits) > 0:
            return possibleSplits[np.argmax([a[1] for a in possibleSplits])]
        else:
            return possibleSplits

    def ValAfterSplit(self, feature, possiblePrices):
        # Creo un log che contiene solo i le reward che contengono la feature
        subLog1 = self.SubLogFeature(feature)
        # Creo il log della feature complementare
        subLog2 = self.SubLogFeature(ComputeComplemetaryFeature(feature))
        if len(self.logRewards) > 0:
            prob1 = len(subLog1) / len(self.logRewards)
            prob2 = len(subLog2) / len(self.logRewards)
        else:
            prob1 = 0
            prob2 = 0

        # Tengo il numero totale di persone in comune hai due learner
        t = self.learner.t

        # Creo due learner trainati con i sublog delle feature
        learn1 = self.LearnerSubContext(subLog1)
        learn2 = self.LearnerSubContext(subLog2)

        # Imposto t
        learn1.t = t
        learn2.t = t

        # Calcolo il miglior valore dei due subLearner
        expectedValue1 = learn1.OptimalExpectedValue(possiblePrices)
        expectedValue2 = learn2.OptimalExpectedValue(possiblePrices)

        # Ritorno la probabilità del primo learner * il suo valore migliore + stessa cosa per il secondo
        ris = [prob1 * expectedValue1 + prob2 * expectedValue2, learn1, learn2]
        return ris


class ContextManager:
    def __init__(self, nArms, featureSpace, categories, week=-1, contextKnown=False):
        self.nArms = nArms
        self.featureSpace = featureSpace

        if contextKnown:
            # Se conosco il context
            self.featuresContext = {categories[i]: i for i in range(len(categories))}
            self.contextsSet = {}
        else:
            # Altrimenti inizio con un unico context
            self.featuresContext = {self.featureSpace[i]: 0 for i in range(len(featureSpace))}
            self.contextsSet = {0: Context(0, featureSpace, TSLearner(nArms), [])}

        self.week = week
        self.time = 0

    def AddContext(self, subspace):
        # Creo un context che gestisce solo uno subspace
        contextID = len(self.contextsSet)
        self.contextsSet[contextID] = Context(contextID, subspace, TSLearner(self.nArms))
        for t in subspace:

            # Aggiorno chi gestisce la specifica feature
            self.featuresContext[t] = contextID

    def SelectArm(self, personType, time, possiblePrices):
        self.time = time
        self.Split(self.time, possiblePrices)

        # Viene chiamato il learner del context che gestisce quel tipo di persona
        return self.contextsSet[self.featuresContext[personType]].learner.PullArm(possiblePrices)

    def Split(self, time, possiblePrices):
        if (self.week != -1) and ((time + 1) % self.week == 0):
            contextsSetCopy = self.contextsSet.copy()
            for index, context in self.contextsSet.items():
                split = context.Split(possiblePrices)
                if split:

                    # Recupero le informazioni dello split migliore
                    feature = split[0]
                    learner1 = split[2]
                    learner2 = split[3]
                    number = len(contextsSetCopy.items())
                    subFeature1 = [x for x in context.subspace if feature in x]
                    subFeature2 = [x for x in context.subspace if feature not in x]
                    subLog1 = context.SubLogFeature(feature)
                    subLog2 = context.SubLogFeature(ComputeComplemetaryFeature(feature))

                    # Creo due nuovi context che gestiranno solo la nuova suddivisione di feature
                    # Uno sostituisce il context che viene diviso l'altro viene messo in coda
                    contextsSetCopy[number] = Context(number, subFeature1, learner1, subLog1)
                    contextsSetCopy[index] = Context(index, subFeature2, learner2, subLog2)
                    self.contextsSet = contextsSetCopy

                    print("Context split con successo a t= " + str(self.time + 1) + ", numero di contexts: " + str(
                        len(self.contextsSet)))
                    # Aggiorno gli indici che indicano quale context gestisce quale feature
                    for context in self.contextsSet.values():
                        for tup in context.subspace:
                            for key in self.featuresContext.keys():
                                if tup == key:
                                    self.featuresContext[key] = context.contextID
        else:
            pass

    def UpdateContext(self, personType, pulledArm, reward):
        # Aggiorno il learner del context che gestisce la persona
        self.contextsSet[self.featuresContext[personType]].Update(personType, pulledArm, reward)

    def ValAttArm(self, personType, pulledArm, possiblePrices):
        # Viene utilizzato il context che gestisce quella tipologia di persona
        alpha = self.contextsSet[self.featuresContext[personType]].learner.betaParameters[pulledArm][0]
        beta = self.contextsSet[self.featuresContext[personType]].learner.betaParameters[pulledArm][1]
        price = possiblePrices[pulledArm]
        return (alpha / (alpha + beta)) * price
