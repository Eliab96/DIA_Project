import numpy as np


def function(x, s, m):
    return (1 - np.exp(-s * x)) * m


class EnvironmentManager:
    '''
        Structure che contiene tutte le informazioni del progetto
    '''
    def __init__(self):
        self.phaseLabels = ['Working Hours', 'Off Hours', 'Weekend']

        # Indica quale delle 3 sottocampagne viene utilizzata durante le varie mezze giornate
        self.phaseSeq = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2]
        self.featureLabels = ['Employed - Adult', 'Non-employed - Adult', 'Non-employed - Teenager']
        self.sigma = 2.0
        self.clickFunctions = {
            'Employed - Adult':
                [(lambda x, s=1, m=10: function(x, s, m)),
                 (lambda x, s=1, m=80: function(x, s, m)),
                 (lambda x, s=1, m=60: function(x, s, m))],
            'Non-employed - Adult':
                [(lambda x, s=1, m=60: function(x, s, m)),
                 (lambda x, s=1, m=75: function(x, s, m)),
                 (lambda x, s=1, m=70: function(x, s, m))],
            'Non-employed - Teenager':
                [(lambda x, s=1, m=80: function(x, s, m)),
                 (lambda x, s=1, m=20: function(x, s, m)),
                 (lambda x, s=1, m=100: function(x, s, m))]
        }
        self.features = {'Employed': ['e', 'n'], 'Age': ['t', 'a']}
        self.featureSpace = [('e', 't'), ('e', 'a'), ('n', 't'), ('n', 'a')]
        self.categories = [('e', 'a'), ('n', 'a'), ('n', 't')]
        self.prices = [5, 7.5, 10, 12.5, 15, 17.5]
        self.probabilities = [[0.75, 0.71, 0.65, 0.57, 0.45, 0.33],
                              [0.55, 0.42, 0.28, 0.12, 0.00, 0.00],
                              [0.90, 0.80, 0.62, 0.36, 0.20, 0.10]]

    '''
        Restituisce il peso di ogni fase (quante volte quella fase appare/numero totale di fasi durante la settimana)
    '''
    def GetPhaseWeights(self):
        a = np.array(self.phaseSeq)
        _, counts = np.unique(a, return_counts=True)
        mcd = sum(counts)
        return [w / mcd for w in counts]

    '''
        Restituisce le varie fasi ripetute per il sampleFactor
    '''
    def GetPhaseList(self, sampleFactor):
        phaseList = []
        for i in range(len(self.phaseSeq)):
            phaseList += [self.phaseSeq[i]] * sampleFactor
        return phaseList

    def GetIndexedCategories(self):
        return {i: c for i, c in enumerate(self.categories)}


