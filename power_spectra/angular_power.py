from .matter_power import MatterPower
from ..halos.base.profile.profile import Profile
from ..halos.base.mass_func.mass_func import MassFunc
from ..halos.base.clump_mass_func.clump_mass_func import ClumpMassFunc
from ..halos.base.bias.bias import Bias
from ..config.config import Config

import numpy as np
from scipy import interpolate


#speed of light
c = 3e5 #km/s

class Cl:
    def __init__(self,
                 cfg: Config,
                 mass_func: MassFunc,
                 smooth_profile: Profile,
                 bias: Bias,
                 clump_mass_func: ClumpMassFunc,
                 clump_profile: Profile,
                 clump_distribution: Profile
                 ):
        """
        Creates a halo model object
        
        :param cfg: Config object
        :type cfg: Config
        :param mass_func: halo mass fucntion
        :type mass_func: MassFunc
        :param bias: bias
        :type bias: Bias
        :param smooth_profile: smooth profile
        :type smooth_profile: Profile
        :param clump_mass_func: Description
        :type clump_mass_func: ClumpMassFunc
        :param clump_profile: Description
        :type clump_profile: Profile
        :param clump_distribution: Description
        :type clump_distribution: Profile
        """
        
        self.cfg = cfg
        self.mass_func = mass_func
        self.smooth_profile = smooth_profile
        self.bias = bias
        self.clump_mass_func = clump_mass_func
        self.clump_profile = clump_profile
        self.clump_distribution = clump_distribution

        
        z_linspace = np.linspace(cfg.z_min, cfg.z_max, cfg.N_z)
        self.Pm_list = []

        #get matter power spectrum at each z value
        for z in z_linspace:
            self.cfg.z = z
            Pm = MatterPower(
                cfg, 
                mass_func=mass_func, 
                smooth_profile=smooth_profile, 
                bias=bias,
                clump_mass_func=clump_mass_func, 
                clump_profile=clump_profile, 
                clump_distribution=clump_distribution
                )
            self.Pm_list.append(Pm)

        print("interpolating power spectra over k and z")

        lnk_logspace = np.log(np.logspace(np.log10(self.cfg.k_min), np.log10(self.cfg.k_max), self.cfg.N_k))
        z_linspace = np.linspace(self.cfg.z_min, self.cfg.z_max, self.cfg.N_z)

        self.P_1h_ss_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, self.P_1h_ss.T)
        self.P_1h_sc_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, self.P_1h_sc.T)
        self.P_1h_self_c_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, self.P_1h_self_c.T)
        self.P_1h_cc_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, self.P_1h_cc.T)
        
        self.P_2h_ss_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, self.P_2h_ss.T)
        self.P_2h_sc_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, self.P_2h_sc.T)
        self.P_2h_cc_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, self.P_2h_cc.T)

    
    def lensing_kernel(self, z):
        cfg = self.cfg

        H0 = cfg.cosmo.H0 #returns in km/s/Mpc
        Om0 = cfg.cosmo.Om0
        prefactor = 3/2 * Om0 * H0**2 / c**2 
        distance_factor = (cfg.cosmo.comovingDistance(z_max = z)
                            * cfg.cosmo.comovingDistance(z_max = cfg.z_sources - z)
                            / cfg.cosmo.comovingDistance(z_max = cfg.z_sources))
        
        return prefactor * distance_factor * (1+z)

    def Cl_1h_ss(self):



        return 




def w(z, z_source, cosmo:cosmology):
    H0 = cosmo.H0 #returns in km/s/Mpc
    Om0 = cosmo.Om0
    prefactor = 3/2 * Om0 * H0**2 / c**2 
    distance_factor = cosmo.comovingDistance(z_max = z) * cosmo.comovingDistance(z_max = z_source - z) / cosmo.comovingDistance(z_max = z_source)
    #distance_factor = cosmo.comovingDistance(z_max = z) * cosmo.comovingDistance(z_min = z, z_max = z_source) / cosmo.comovingDistance(z_max = z_source)
    return prefactor * distance_factor * (1+z)

def Pm_to_Pk(l, Pm_kz, z_source, cosmo:cosmology):
    """
    Returns convergence power spectrum at value l.
    Args:
        l: angular scale.
        P_k_z: matter matter powerspectrum. Must have only k, z as input; so Pmm = Pmm(k,z).
        p: probability distribution of sources over redshift; p(z). If input p is a scalar, it uses a delta function.
        cosmo: cosmology object from colossus.
        z_Hubble: Furthest redshift source.
    """
    
    def integrand(z):
        D = cosmo.comovingDistance(z_max = z)
        H = cosmo.Hz(z) #returns in km/s/Mpc --> note that c is given in km/s
        return 2*np.pi * c/H * w(z, z_source, cosmo)**2 / D**2 * Pm_kz(l/D, z)

    I, err = integrate.quad(integrand, 0, z_source, epsrel=1e-4)

    return I