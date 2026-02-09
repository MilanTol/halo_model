from abc import ABC, abstractmethod

class Profile(ABC):
    @abstractmethod
    def fourier(self, cosmo, k, M, z):
        pass