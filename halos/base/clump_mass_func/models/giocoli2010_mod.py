from ..clump_mass_func import ClumpMassFunc

import numpy as np

class Giocoli2010_mod(ClumpMassFunc):

    def __init__(self, m0, beta):
        self.m0 = m0
        self.alpha = beta

    def standard(self, m, M, z): 
        #This function is missing c/c_bar, 
        #this is only valid for  fixed relation concentration-mass.
        """Clump mass function from Giocoli et al. (2010)"""
        A = 9.33e-4 
        alpha = -0.9          
        beta = 12.2715
        return M * (1+z)**0.5 * A * m**alpha * np.exp(-beta * (m / M)**3)
    
    def cmf(self, m, M, z):
        return (1 + self.m0/m)**self.beta * self.standard(m, M, z)
    
    
    