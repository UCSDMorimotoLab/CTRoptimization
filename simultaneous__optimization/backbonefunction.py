import numpy as np
from ozone.api import ODEFunction
from backbonesystem import BackboneSystem


class BackboneFunction(ODEFunction):
    def setup(self):
        pass

    def initialize(self, system_init_kwargs=None):
        self.set_system(BackboneSystem, system_init_kwargs)
        k=10
        self.declare_state('R', 'R_dot', shape=(k,3,3), targets=['R'])
        self.declare_parameter('uhat', shape=(k,3,3), targets=['uhat'], dynamic=True)
                 
        