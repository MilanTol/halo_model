from ..r_vir import R_vir

import numpy as np

class rho_m340(R_vir):

    def __init__(self, delta):
        """      
        :param delta: average overdensity of halo as a fraction of 
        background matter densityat z=0 contained within virial radius.
        """
        self.delta = delta
    
    def R_vir(self, cosmo, M):
        rho_bg = cosmo.rho_m(z=0)*1e9 #converting to Mpc

        return (3 * M / (4 * np.pi * self.delta * rho_bg))**(1/3)