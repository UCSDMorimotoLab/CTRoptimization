import numpy as np
from openmdao.api import Group, ExecComp
from bborientation import BborientationComp


class BackboneSystem(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=100, types=int)
        self.options.declare('k', default=1, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        comp1 = BborientationComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('Bborientationcomp',comp1, promotes=['*'])



        






