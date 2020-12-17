import numpy as np
import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTSLearner:
    def __init__(self, arms):
        self.arms = arms
        self.nArms = len(arms)
        self.collectedRewards = np.array([])
        self.pulledArms = []
        self.means = np.zeros(self.nArms)
        self.sigmas = np.ones(self.nArms) * 10
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                                                                    normalize_y=False, n_restarts_optimizer=9)

    def Update(self, pulledArm, reward):
        # Salvo la reward
        self.collectedRewards = np.append(self.collectedRewards, reward)
        self.pulledArms.append(self.arms[pulledArm])

        # Aggiorno il modello in base alla reward
        x = np.atleast_2d(self.pulledArms).T
        y = self.collectedRewards
        self.gp.fit(x, y)

        # Il GPR aggiornerà tutti i valori di media e sigma in base al valore aggiunto con il fit
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def PullArm(self, armId):
        # Utilizzo una distribuzione normale per decidere l'arm
        sampledValue = np.random.normal(self.means[armId], self.sigmas[armId])
        sampledValue = np.maximum(0, sampledValue)
        return sampledValue

    '''
        Ritorna l'indice dell'arm cercato
    '''

    def FindArm(self, arm):
        for idx in range(self.nArms):
            if self.arms[idx] == arm:
                return idx
        return False

    '''
        Faccio il fit per gli hyperparameters
    '''

    def Hyper(self, samples):
        x = np.atleast_2d(samples[0]).T
        y = [y for (x, y) in zip(samples[0], samples[1])]
        self.gp.fit(x, y)
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=self.gp.kernel, alpha=self.gp.alpha,
                                                                    normalize_y=False, n_restarts_optimizer=0)


'''
    Variante con Sliding windows
'''


class SWSubCampaignLearner(GPTSLearner):
    def __init__(self, arms, label, windowSize):
        super().__init__(arms)
        self.label = label
        self.windowSize = windowSize

    def PullArms(self):
        # Distribuzione normale per il pull
        sampledValues = np.random.normal(self.means, self.sigmas)
        # Elimino valori negativi ponendoli = 0
        sampledValues = np.maximum(0, sampledValues)
        return sampledValues

    def Update(self, pulledArm, reward, t):
        self.collectedRewards = np.append(self.collectedRewards, reward)
        self.pulledArms.append(self.arms[pulledArm])

        # Tengo salvati solo t valori dipendenti dal valore della sliding window
        if t <= self.windowSize:
            x = np.atleast_2d(self.pulledArms).T
            y = self.collectedRewards
        else:

            # Slide
            x = np.atleast_2d(self.pulledArms[t - self.windowSize:]).T
            y = self.collectedRewards[t - self.windowSize:]
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)

        # Tengo un valore minimo di sigma
        self.sigmas = np.maximum(self.sigmas, 1e-2)


class SubCampaignLearner(GPTSLearner):
    def __init__(self, arms, label):
        super().__init__(arms)
        self.label = label

    def PullArms(self):
        # Estraggo il valore della distribuzione per ogni arms
        sampledValues = np.random.normal(self.means, self.sigmas)

        # Elimino valori negativi
        sampledValues = np.maximum(0, sampledValues)
        return sampledValues
