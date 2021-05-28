import numpy as np
from ozone.api import ODEFunction
from CtrSystem import CtrSystem
from kinematics_comp import KinematicsComp
from s_comp import SComp
from rhs_comp import RHSComp

class CtrFunction(ODEFunction):
    def setup(self):
        pass


    def initialize(self, system_init_kwargs=None):
        
        # num_nodes = self.options['num_nodes']
        # self.options.declare('k', default=1, types=int)
        self.set_system(CtrSystem, system_init_kwargs)
        # k=system_init_kwargs['k']
        k=1
        self.declare_state('psi', 'psi_dot', shape=(k,3), targets=['psi'])
        self.declare_state('dpsi_ds', 'dpsi_ds_dot', shape=(k,3), targets=['dpsi_ds'])
        
         

        
        self.declare_parameter('K_out', shape=(k,3,3), targets=['K_out'], dynamic=True)
        
        