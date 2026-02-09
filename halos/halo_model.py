from .base.profile.profile import Profile
from .base.mass_func.mass_func import MassFunc
from .base.clump_mass_func.clump_mass_func import ClumpMassFunc

from scipy import integrate
import numpy as np

class HaloModel:

    def __init__(self,
                 cosmo, 
                 mass_func: MassFunc,
                 smooth_profile: Profile,
                 clump_mass_func: ClumpMassFunc,
                 clump_profile: Profile,
                 clump_distribution: Profile
                 ):
        
        self.mass_func = mass_func
        self.smooth_profile = smooth_profile
        self.clump_mass_func = clump_mass_func
        self.clump_profile = clump_profile
        self.clump_distribution = clump_distribution


    def P_1h_ss(self, z=0, M_min = 1, M_max = 1e16, m_min=1):

        rho0 = self.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):
            M = np.exp(ln_M)
            M_smooth = (1 - self.clump_mass_func.f(M, z, m_min=m_min)) * M
            n = self.mass_func(M, z)

            first_term = prefactor * n
            second_term = M_smooth**2 * self.smooth_profile()


    