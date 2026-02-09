from ..clump_mass_func import ClumpMassFunc

import numpy as np

class Giocoli2010(ClumpMassFunc):

    def _cmf(self, m, M_parent, z): 
        #This function is missing c/c_bar, 
        #this is only valid for  fixed relation concentration-mass.
        """Clump mass function from Giocoli et al. (2010)"""
        A = 9.33e-4 
        alpha = -0.9          
        beta = 12.2715
        return M_parent * (1+z)**0.5 * A * m**alpha * np.exp(-beta * (m / M_parent)**3)
    
    
    