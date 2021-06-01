import numpy as np
from ozone.api import ODEFunction


class BackboneFunction(ODEFunction):
    def setup(self):
        pass

    def initialize(self, system_init_kwargs=None):
        self.set_system(BackboneSystem, system_init_kwargs)
        system_init_kwargs = dict(
            k=k,
        )
        
        self.declare_state('R', 'R_dot', shape=(k,3,3), targets=['R'])
        self.declare_parameter('uhat', shape=(k,3,3), targets=['uhat'], dynamic=True)
                 
        