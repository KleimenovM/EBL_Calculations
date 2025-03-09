import numpy as np


class SpectralModel:
    def __init__(self, name):
        self.name = name
        return

    def __call__(self, energy):
        return


class PowerLaw(SpectralModel):
    def __init__(self, name, phi0, e0, gamma):
        super().__init__(name)
        self.phi0 = phi0
        self.e0 = e0
        self.gamma = gamma

    def __call__(self, energy):
        return self.phi0 * (energy / self.e0) ** (-self.gamma)


class LogParabola(SpectralModel):
    def __init__(self, name, phi0, e0, gamma, beta):
        super().__init__(name)
        self.phi0 = phi0
        self.e0 = e0
        self.gamma = gamma
        self.beta = beta

    def __call__(self, energy):
        ee0 = energy / self.e0
        return self.phi0 * (ee0 ** (-self.gamma - self.beta * np.log(ee0)))


class ExpCutoff(SpectralModel):
    def __init__(self, name, phi0, e0, gamma, ecutoff):
        super().__init__(name)
        self.phi0 = phi0
        self.e0 = e0
        self.gamma = gamma
        self.ecutoff = ecutoff

    def __call__(self, energy):
        return self.phi0 * (energy / self.e0) ** (-self.gamma) * np.exp(-energy / self.ecutoff)


class LogParabolaCutoff(SpectralModel):
    def __init__(self, name, phi0, e0, gamma, beta, ecutoff):
        super().__init__(name)
        self.phi0 = phi0
        self.e0 = e0
        self.gamma = gamma
        self.beta = beta
        self.ecutoff = ecutoff

    def __call__(self, energy):
        ee0 = energy / self.e0
        return self.phi0 * (ee0 ** (-self.gamma - self.beta * np.log(ee0))) * np.exp(-energy / self.ecutoff)


class GreauxModel(SpectralModel):
    def __init__(self, name, phi0, e0, gamma, beta, eta, lam):
        super().__init__(name)
        self.phi0 = phi0
        self.e0 = e0
        self.gamma = gamma
        self.beta = beta
        self.eta = eta
        self.lam = lam

    def __call__(self, energy):
        ee0 = energy / self.e0
        degree = self.eta - self.gamma * np.log(ee0) - self.beta * np.log(ee0)**2
        return self.phi0 * np.exp(degree) * np.exp(-self.lam * ee0)

    def get(self, energy, gamma, beta, eta, lam):
        ee0 = energy / self.e0
        degree = eta - gamma * np.log(ee0) - beta * np.log(ee0)**2
        return self.phi0 * np.exp(degree) * np.exp(-lam * ee0)
