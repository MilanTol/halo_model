from .base.profile.profile import Profile
from .base.mass_func.mass_func import MassFunc
from .base.clump_mass_func.clump_mass_func import ClumpMassFunc
from ..config.config import Config

from scipy import integrate
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy import interpolate


class HaloModel:

    def __init__(self,
                 cfg: Config,
                 mass_func: MassFunc,
                 smooth_profile: Profile,
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
        self.clump_mass_func = clump_mass_func
        self.clump_profile = clump_profile
        self.clump_distribution = clump_distribution


        ######################################################################################
        # To speed up computation we will interpolate Ic and Jc integrals in the initializer
        ######################################################################################

        # Worker function to compute both Ic and Jc for one (k, M)
        def compute_point(args):
            k, M = args
            Ic_val = self.Ic_analytic(k, M)
            Jc_val = self.Jc_analytic(k, M)
            return (k, M, Ic_val, Jc_val)

        print("interpolating Ic and Jc functions...")

        # Define grids
        k_grid = np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.N_k)  
        M_grid = np.logspace(np.log10(cfg.M_min), np.log10(cfg.M_max), cfg.N_M)   

        # Allocate arrays
        Ic_vals = np.zeros((len(k_grid), len(M_grid)))
        Jc_vals = np.zeros((len(k_grid), len(M_grid)))
    
        args = [(k, M) for k in k_grid for M in M_grid]

        with Pool(processes=cpu_count()) as pool:
            points = pool.map(compute_point, args)

        # Fill results into arrays
        for (k, M, Ic_val, Jc_val) in points:
            i = np.where(k_grid == k)[0][0]
            j = np.where(M_grid == M)[0][0]
            Ic_vals[i, j] = Ic_val
            Jc_vals[i, j] = Jc_val

        # Create interpolators
        self.Ic = interpolate.RegularGridInterpolator((k_grid, M_grid), Ic_vals,
                                                        bounds_error=False, fill_value=None)
        self.Jc = interpolate.RegularGridInterpolator((k_grid, M_grid), Jc_vals,
                                                        bounds_error=False, fill_value=None)


    def Ic_analytic(self, k, M):
        """I_c integral, eq. (30) in Giocoli et al..
        Args:
            k: wavenumber [Mpc/h]^-1
            M_parent: total halo mass [h M_sun]
        """
        
        def m_integrand(lnm):
            m = np.exp(lnm)
            clump_profile_temp = self.clump_profile.fourier(self.cfg.cosmo, k, m, self.cfg.z)
            return (1/M 
                    * clump_profile_temp 
                    * self.clump_mass_func(m, M, self.cfg.z) 
                    * m) # Jacobian for dlnm to dm conversion 

        I, error = integrate.quad(m_integrand, np.log(self.cfg.m_min), np.log(M), epsrel=1e-4, limit=200)

        return I


    def Jc_analytic(self, k, M):
        """J_c integral, eq. (31) in Giocoli et al..
        Args:
            k: wavenumber [Mpc/h]^-1
            M_parent: total halo mass [h M_sun]
        """

        def m_integrand(lnm):
            m = np.exp(lnm)
            return (m*(1/M)**2 
                    * self.clump_profile.fourier(self.cfg.cosmo, k, m, self.cfg.z)**2  
                    * self.clump_mass_func(m, M, self.cfg.z) 
                    * m)   # Jacobian for dlnm to dm conversion
        
        J, error = integrate.quad(m_integrand, np.log(self.cfg.m_min), np.log(M), epsrel=1e-4, limit=200)

        return J


    def P_1h_ss(self, k):
        cfg = self.cfg
        
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):
            M = np.exp(ln_M)
            M_smooth = (1 - self.clump_mass_func.f(M, cfg.z, m_min=cfg.m_min)) * M
            n = self.mass_func(M, cfg.z)

            first_term = prefactor * n
            second_term = M_smooth**2 * self.smooth_profile.fourier(cfg.cosmo, k, M, cfg.z)**2

            return first_term*second_term*M #times M for jacobian dlnM to dM

        I, error = integrate.quad(M_integrand, np.log(cfg.M_min), np.log(cfg.M_max), limit=200, epsrel=1e-4)
        return I
    

    def P_1h_sc(self, k):
        """
        Docstring for P_1h_sc
        
        :param self: Description
        :param k: Description
        :param z: Description
        :param M_min: Description
        :param M_max: Description
        :param m_min: Description
        """
        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):
            M = np.exp(ln_M) 
            n = self.mass_func(M, cfg.z)
            first_term = 2 * prefactor * M  * n

            M_smooth = (1 - self.clump_mass_func.f(M, cfg.z, cfg.m_min)) * M  # Smooth mass component
            second_term = M_smooth * self.smooth_profile.fourier(cfg.cosmo, k, M, cfg.z)

            third_term = self.clump_distribution.fourier(self.cfg.cosmo, k, M, cfg.z) * self.Ic((k, M))

            return first_term * second_term * third_term * M # Jacobian for dlnM to dM conversion

        I, error = integrate.quad(M_integrand, np.log(cfg.M_min), np.log(cfg.M_max), epsrel=1e-4, limit=200)

        return I
        

    def P_1h_self_c(self, k):
        """self-clump component for the 1-halo term. See eq. (27) Giocoli et al.. 
        Args:
            k: wavenumber [Mpc/h]^-1
        """
        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):

            M = np.exp(ln_M)
            n = self.mass_func(M, cfg.z)

            first_term = M**2 * prefactor * n
            return first_term * self.Jc((k, M)) * M # Jacobian for dlnM to dM conversion   

        I, error = integrate.quad(M_integrand, np.log(cfg.M_min), np.log(cfg.M_max), epsrel=1e-4, limit=200)

        return I


    def P_1h_cc(self, k):
        """clump-clump component for the 1-halo term. See eq. (28) Giocoli et al.. 
        Args:
            k: wavenumber [Mpc/h]^-1
        """
        cfg = self.cfg
        rho0 = cfg.cosmo.rho_m(z=0) * 1e9 #should be at z=0!
        prefactor = 1/rho0**2

        def M_integrand(ln_M):

            M = np.exp(ln_M)
            n = self.mass_func(M, cfg.z)

            first_term = M**2 * prefactor * n
            second_term = self.clump_distribution.fourier(cfg.cosmo, k, M, cfg.z)**2 * self.Ic((k, M))**2

            return first_term * second_term * M # Jacobian for dlnM to dM conversion
        
        I, error = integrate.quad(M_integrand, np.log(cfg.M_min), np.log(cfg.M_max), epsrel=1e-4, limit=200)

        return I
    
    
    def P_1h(self ,k):
        """returns the 1-halo power spectrum
        Args:
            k: wavenumber [Mpc/h]^-1
        """
        return self.P_1h_ss(k) + self.P_1h_sc(k) + self.P_1h_self_c(k) + self.P_1h_cc(k)
    
