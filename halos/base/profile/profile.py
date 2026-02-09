from abc import ABC, abstractmethod

class Profile(ABC):
    @abstractmethod
    def fourier(self, cosmo, k, M, z):
        """
        Docstring for fourier
        
        :param cosmo: cosmology object
        :param k: wavenumber k (h/Mpc)
        :param M: halo mass
        :param z: redshift 
        """
        pass