from ..mass_func import MassFunc
from colossus.lss.mass_function import massFunction

class Sheth1999(MassFunc):
    def hmf(self, M, z):
        return massFunction(M, z=z, model = 'sheth99', q_out='dndlnM') / M