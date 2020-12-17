class ExperimentManager:
    def __init__(self, personManager, contextManager, environment):
        self.personManager = personManager
        self.contextManager = contextManager
        self.environment = environment
        self.rewardLog = []
        self.possiblePrices = self.environment.possiblePrices
        self.expectedValues = [[c * 0.5 for c in self.possiblePrices] for cat in range(3)]

    def PlayExperiment(self, numPersons):
        countSucc = 0
        # Per ogni persona il context che la gestisce propone un prezzo e se essa accetta la reward aumenta
        for countPerson in range(numPersons):
            possiblePrices = self.environment.possiblePrices
            personType = self.personManager.AddPerson()
            personFeatures = self.personManager.types[personType]

            # Seleziono un possibile prezzo per quella tipologia di persona
            pulledArm = self.contextManager.SelectArm(personFeatures, countPerson, possiblePrices)

            # Calcolo quale sarebbe stata la possibile reward
            expectedValArm = self.contextManager.ValAttArm(personFeatures, pulledArm, possiblePrices)

            # Vedo se la persona accetta o rifiuta la proposta
            rewardPerson = self.environment.Round(personType, pulledArm)

            if rewardPerson > 0: countSucc = countSucc + 1

            # Aggiorno il context e salvo il risultato
            self.contextManager.UpdateContext(personFeatures, pulledArm, rewardPerson)
            self.rewardLog.append([personType, pulledArm, rewardPerson, expectedValArm])
        print(str(countSucc)+" successi su "+str(numPersons) +
              " Con: " + str(len(self.contextManager.contextsSet)) + " contexts")
        return self.rewardLog

    def RunPricingExperiment(self, nCategoriesClicks):
        for index, clicks in enumerate(nCategoriesClicks):
            featurePerson = self.personManager.types[index]

            # Individuo quante persone cliccano sull'add
            roundedClicks = int(round(clicks))
            for n in range(roundedClicks):
                # Vedo quante delle persone che hanno cliccato effettivamente comprano il prodotto
                pulledArm = self.contextManager.SelectArm(featurePerson, n, self.possiblePrices)
                rewardPerson = self.environment.Round(index, pulledArm)
                self.contextManager.UpdateContext(featurePerson, pulledArm, rewardPerson)
                self.rewardLog.append([index, pulledArm, rewardPerson])

            idx = self.contextManager.featuresContext[featurePerson]

            # Calcolo l'expected reward per quella tipologia di persona per ogni possibile prezzo
            for c in range(len(self.possiblePrices)):
                self.expectedValues[idx][c] = self.contextManager.contextsSet[idx]. \
                    learner.ExpectedValue(c, self.possiblePrices[c])
