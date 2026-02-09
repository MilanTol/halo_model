from ..mass_func.mass_func import MassFunc

from abc import ABC, abstractmethod
from scipy import integrate
import numpy as np

class ClumpMassFunc(ABC):

    @abstractmethod
    def _cmf(self, m, M, z):
        """Implementation of the clump mass function"""
        pass


    def f(self, M, z, m_min=1):
        """
        returns fraction of mass contained in clumps in halo of mass M
        
        :param M: parent halo mass M (total mass, not just smooth component!)
        :param z: Redshift
        :param m_min: minimum clump mass to consider (standard=1)
        """

        def integrand(ln_m):
            m = np.exp(ln_m)
            return self._cmf(m, M, z) * m
        
        mass_in_clumps, error = integrate.quad(integrand, np.log(m_min), np.log(M))
        return mass_in_clumps / M


    def __call__(self, m, M, z):
        """ Returns the clump mass function for input parameters.

        Args:
            m (:obj:`float` or `array`): clump mass
            M (:obj:`float` or `array`): parent halo mass (total mass, not just smooth component!).
            z (:obj:`float`): redshift.

        Returns:
            (:obj:`float` or `array`): clump mass function 
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
        """
        return self._cmf(m=m, M=M, z=z)