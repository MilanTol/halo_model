from .matter_power import MatterPower
from ..halos.base.profile.profile import Profile
from ..halos.base.mass_func.mass_func import MassFunc
from ..halos.base.clump_mass_func.clump_mass_func import ClumpMassFunc
from ..halos.base.bias.bias import Bias
from ..config.config import Config

import numpy as np
from scipy import interpolate
from scipy import integrate

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

        #first calculate all power spectrum component values and store them in arrays:
        lnk_logspace = np.log(np.logspace(np.log10(self.cfg.k_min), np.log10(self.cfg.k_max), self.cfg.N_k))
        P_1h_ss = []
        P_1h_sc = []
        P_1h_self_c = []
        P_1h_cc = []
        P_2h = []

        for Pm in self.Pm_list:
            P_1h_ss.append([Pm.P_1h_ss(np.exp(lnk)) for lnk in lnk_logspace])
            P_1h_sc.append([Pm.P_1h_sc(np.exp(lnk)) for lnk in lnk_logspace])
            P_1h_self_c.append([Pm.P_1h_self_c(np.exp(lnk)) for lnk in lnk_logspace])
            P_1h_cc.append([Pm.P_1h_cc(np.exp(lnk)) for lnk in lnk_logspace])
            P_2h.append([Pm.P_2h(np.exp(lnk)) for lnk in lnk_logspace])
            
        #interpolate over logk and z:
        self.P_1h_ss_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, np.array(P_1h_ss).T)
        self.P_1h_sc_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, np.array(P_1h_sc).T)
        self.P_1h_self_c_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, np.array(P_1h_self_c).T)
        self.P_1h_cc_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, np.array(P_1h_cc).T)
        self.P_2h_lnk = interpolate.RectBivariateSpline(lnk_logspace, z_linspace, np.array(P_2h).T)
    

    #matter power spectrum components interpolated over k and z:
    def P_1h_ss(self, k, z):
        return self.P_1h_ss_lnk(np.log(k), z)
    def P_1h_sc(self, k, z):
        return self.P_1h_sc_lnk(np.log(k), z)
    def P_1h_self_c(self, k, z):
        return self.P_1h_self_c_lnk(np.log(k), z)
    def P_1h_cc(self, k, z):
        return self.P_1h_cc_lnk(np.log(k), z)
    def P_1h(self, k, z):
        return self.P_1h_ss(k, z) + self.P_1h_sc(k, z) + self.P_1h_self_c(k,z) + self.P_1h_cc(k,z)
    def P_2h(self, k, z):
        return self.P_2h_lnk(np.log(k), z)
    

    def lensing_kernel(self, z):
        cfg = self.cfg

        H0 = cfg.cosmo.H0 #returns in km/s/Mpc
        Om0 = cfg.cosmo.Om0
        prefactor = 3/2 * Om0 * H0**2 / c**2 
        distance_factor = (cfg.cosmo.comovingDistance(z_max = z)
                            * cfg.cosmo.comovingDistance(z_max = cfg.z_sources - z)
                            / cfg.cosmo.comovingDistance(z_max = cfg.z_sources))
        
        return prefactor * distance_factor * (1+z)


    def P_to_C(self, l, P, w):
        """
        Returns convergence power spectrum at value l using lensing kernel.
        Args:
            l: angular scale.
            P: 3d powerspectrum. Must have only k, z as arguments; so P = P(k,z).
            w: weight function. Must have only z as arguments: so w = w(z)
            cosmo: cosmology object from colossus.
        """
        cfg = self.cfg

        def integrand(z):
            D = cfg.cosmo.comovingDistance(z_max = z)
            H = cfg.cosmo.Hz(z) #returns in km/s/Mpc --> note that c is given in km/s
            return 2*np.pi * c/H * w(z)**2 / D**2 * P(l/D, z)

        I, err = integrate.quad(integrand, 0, cfg.z_sources, limit=200, epsrel=1e-4)

        return I
    

    def Cl_1h_ss(self, l):
        return self.P_to_C(self, l, self.P_1h_ss, self.lensing_kernel)
    def Cl_1h_sc(self, l):
        return self.P_to_C(self, l, self.P_1h_sc, self.lensing_kernel)
    def Cl_1h_self_c(self, l):
        return self.P_to_C(self, l, self.P_1h_self_c, self.lensing_kernel)
    def Cl_1h_cc(self, l):
        return self.P_to_C(self, l, self.P_1h_cc, self.lensing_kernel)
    def Cl_1h(self, l):
        return self.P_to_C(self, l, self.P_1h, self.lensing_kernel)
    def Cl_2h(self, l):
        return self.P_to_C(self, l, self.P_2h, self.lensing_kernel)
    