from abc import ABC, abstractmethod

class Profile(ABC):
    @abstractmethod
    def _fourier(self, cosmo, k, M, z):
        """
        Docstring for fourier
        
        :param cosmo: cosmology object
        :param k: wavenumber k (h/Mpc)
        :param M: halo mass
        :param z: redshift 
        """
        pass

    def fourier(self, cosmo, k, M, z):
        """
        Docstring for fourier
        
        :param cosmo: cosmology object
        :param k: wavenumber k (h/Mpc)
        :param M: halo mass
        :param z: redshift 
        """
        return self._fourier(cosmo, k, M, z)