import random


class PersonManager:
    def __init__(self, types, probabilities, features):
        self.types = types
        self.nTypes = len(self.types)
        self.probabilities = probabilities
        self.features = features
        self.personCount = 0
        self.typesCount = [0] * self.nTypes

    def AddPerson(self):
        personType = random.randint(0, self.nTypes - 1)
        self.personCount += 1
        self.typesCount[personType] += 1
        return personType
