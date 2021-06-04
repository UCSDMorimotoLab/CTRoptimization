import numpy as np
from ozone.api import ODEFunction
from ctr_framework.CtrSystem import CtrSystem

class CtrFunction(ODEFunction):
    def setup(self):
        pass


    def initialize(self, k):
        
        # num_nodes = self.options['num_nodes']
        # self.options.declare('k', default=1, types=int)
        system_init_kwargs = dict(
            k=k,
        )
        self.set_system(CtrSystem, system_init_kwargs)
        self.declare_state('psi', 'psi_dot', shape=(k,3), targets=['psi'])
        self.declare_state('dpsi_ds', 'dpsi_ds_dot', shape=(k,3), targets=['dpsi_ds'])
        
        self.declare_parameter('K_out', shape=(k,3,3), targets=['K_out'], dynamic=True)
        
        