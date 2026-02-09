from ..profile import Profile
from ...concentration.concentration import Concentration
from ...r_vir.r_vir import R_vir
from .nfw_helper_functions.Si_Ci_integrals import Si, Ci

import numpy as np

class NFW(Profile):
    def __init__(self, concentration: Concentration, r_vir: R_vir):
        self.c = concentration
        self.r_vir = r_vir
    

    def f(c):
        return 1/(np.log(1 + c) - c / (1 + c))
    

    def _fourier(self, cosmo, k, M, z):
        c = self.c(cosmo, M, z)

        ka = k * self.r_vir(cosmo, M, z) / c

        temp = self.f(c)* (np.sin(ka) * (Si((1 + c)*ka) - Si(ka)) 
                - np.sin(c*ka) / ((1 + c)*ka) 
                + np.cos(ka)*(Ci((1 + c)*ka) - Ci(ka)) )

        return temp
    
