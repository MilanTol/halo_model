import yaml
import os
from colossus.cosmology import cosmology

class Config:

    def __init__(self, path_to_config=None):
        if path_to_config is None:
            path_to_config = "/home/milan/Desktop/thesis/code/config/config.yaml"

        #open the matter power spectrum config file
        with open(path_to_config) as file:
            config = yaml.safe_load(file.read())

        # Matter power spectrum
        self.z          = float(config['z'])
        self.m_min      = float(config['m_min'])
        self.M_min      = float(config['M_min'])
        self.M_max      = float(config['M_max'])
        self.N_M        = int(config['N_M'])
        self.k_min      = float(config['k_min'])
        self.k_max      = float(config['k_max'])
        self.N_k        = int(config['N_k'])
        self.Delta_vir  = int(config['Delta_vir'])
        self.d_vir      = float(config['d_vir'])

        # Convergence power spectrum
        self.z_min      = float(config['z_min'])
        self.z_max      = float(config['z_max'])
        self.z_sources  = float(config['z_sources'])
        self.N_z        = int(config['N_z'])
        self.l_min      = float(config['l_min'])
        self.l_max      = float(config['l_max'])
        self.N_l        = int(config['N_l'])

        # Cosmology parameters
        self.cosmo_params = config['cosmo_params']

        cosmology.addCosmology('myCosmo', **self.cosmo_params)
        self.cosmo = cosmology.setCosmology('myCosmo')

        # halo mass function parameters
        self.M_pivot    = float(config['M_pivot'])
        self.alpha      = float(config['alpha'])

        # clump mass function parameters
        self.m_pivot    = float(config['m_pivot'])
        self.alpha_clump = float(config['alpha_clump'])

        #data storage
        self.Pm_dir     = str(config['Pm_dir'])
        self.Pk_dir     = str(config['Pk_dir'])

    def save(self, path):
        with open(os.path.join(path, 'config.yaml'), 'w') as f:
            yaml.dump(self.to_dict(), f)
    
    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in ['cosmo']
        }

