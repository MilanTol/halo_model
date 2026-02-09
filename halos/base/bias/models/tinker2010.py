from ..bias import Bias
from colossus.lss.bias import haloBias


class BiasTinker2010(Bias):

    def __init__(self, delta_vir):
        self.delta_vir = delta_vir

    def _bias(self, M, z):
        return haloBias(M, z=z, mdef = f'{self.delta_vir}m', model='tinker10')