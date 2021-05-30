import numpy as np
from ozone.api import ODEFunction
from backboneptssystem import Backboneptssystem


class BackboneptsFunction(ODEFunction):
    def setup(self):
        pass

    def initialize(self, system_init_kwargs=None):
        system_init_kwargs = dict(
            k=k,
        )
        self.set_system(Backboneptssystem, system_init_kwargs)
        
        self.declare_state('p', 'p_dot', shape=(k,3,1))
        self.declare_parameter('R', shape=(k,3,3), targets=['R'], dynamic=True)

                 
        