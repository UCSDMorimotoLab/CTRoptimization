import numpy as np
from openmdao.api import Group, ExecComp
from ctr_framework.kinematics_comp import KinematicsComp
from ctr_framework.s_comp import SComp
from ctr_framework.rhs_comp import RHSComp



class CtrSystem(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=2, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        comp1 = SComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('Scomp',comp1, promotes=['*'])
        comp2 = RHSComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('RHScomp',comp2, promotes=['*'])
        comp3= KinematicsComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('kinematicscomp', comp3, promotes=['*'])


        






