from ..concentration import Concentration

class ConcentrationNeto2007(Concentration):
    
    def _concentration(self, cosmo, M, z):#Model C1
        """concentration-mass relation from Neto et al. (2007) 
        from the aquarius simulation, related to M_200"""

        c0 = 4.67
        beta = -0.11
        M_pivot = 1e14  # in M_sun/h

        return c0 * (M / M_pivot)**beta * (1 + z)**-1
    

    

