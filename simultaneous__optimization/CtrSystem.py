import numpy as np
from openmdao.api import Group, ExecComp
from kinematics_comp import KinematicsComp
from s_comp import SComp
from rhs_comp import RHSComp
from bc_comp import BcComp
from  obj_comp import ObjComp
from penalize_comp import PenalizeComp
from initpsi_comp import InitialpsiComp

class CtrSystem(Group):

    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=10, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        comp1 = SComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('Scomp',comp1, promotes=['*'])
        comp2 = RHSComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('RHScomp',comp2, promotes=['*'])
        comp3= KinematicsComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('kinematicscomp', comp3, promotes=['*'])


        






