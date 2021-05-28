import numpy as np
from openmdao.api import Group, ExecComp
from bbpoints_comp import BbpointsComp


class Backboneptssystem(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=10, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        comp1 = BbpointsComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('Bbpointscomp',comp1, promotes=['*'])



        






