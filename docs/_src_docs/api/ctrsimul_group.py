import numpy as np
# import scipy.io
# import openmdao.api as om 
# from openmdao.api import Problem, Group, ExecComp, IndepVarComp, ScipyOptimizeDriver, pyOptSparseDriver
# from openmdao.api import Problem, Group, ExecComp, IndepVarComp, ScipyOptimizeDriver
# from ozone.api import ODEIntegrator
from ctr_framework.stiffness_comp import StiffnessComp
from ctr_framework.CtrFunction import CtrFunction
from ctr_framework.tensor_comp import TensorComp
from ctr_framework.rhs_comp import RHSComp
from ctr_framework.kinematics_comp import KinematicsComp
from ctr_framework.k_comp import KComp
from ctr_framework.sumk_comp import SumkComp
from ctr_framework.sumkm_comp import SumkmComp
from ctr_framework.invsumk_comp import InvsumkComp
from ctr_framework.tubeends_comp import TubeendsComp
from ctr_framework.initpsi_comp import InitialpsiComp
from ctr_framework.penalize_comp import PenalizeComp
from ctr_framework.interpolationkp_comp import InterpolationkpComp
from ctr_framework.straightends_comp import StraightendsComp
from ctr_framework.kappa_comp import KappaComp
from ctr_framework.kout_comp import KoutComp
from ctr_framework.interpolationkb_comp import InterpolationkbComp
from ctr_framework.interpolationkp_comp import InterpolationkpComp
'backbone comps'
from ctr_framework.backbonefunction import BackboneFunction
from ctr_framework.initR_comp import InitialRComp
from ctr_framework.u1_comp import U1Comp
from ctr_framework.u2_comp import U2Comp
from ctr_framework.u3_comp import U3Comp
from ctr_framework.u_comp import UComp
from ctr_framework.uhat_comp import UhatComp
from ctr_framework.bborientation import BborientationComp
from ctr_framework.backboneptsFunction import BackboneptsFunction
'Integrator'
from ctr_framework.finaltime_comp import FinaltimeComp
'base angle'
from ctr_framework.baseangle_comp import BaseangleComp
from ctr_framework.rotp_comp import RotpComp
'constraints'
from ctr_framework.diameter_comp import DiameterComp
from ctr_framework.tubeclearance_comp import TubeclearanceComp
from ctr_framework.bc_comp import BcComp
from ctr_framework.desiredpoints_comp import DesiredpointsComp
from ctr_framework.deployedlength_comp import DeployedlengthComp
from ctr_framework.beta_comp import BetaComp
from ctr_framework.tubestraight_comp import TubestraightComp
from ctr_framework.tiporientation_comp import TiporientationComp
from ctr_framework.baseplanar_comp import BaseplanarComp
from ctr_framework.maxcurvature_comp import MaxcurvatureComp
from ctr_framework.chi_comp import ChiComp
from ctr_framework.gamma_comp import GammaComp
from ctr_framework.kappaeq_comp import KappaeqComp
from ctr_framework.strain_comp import StrainComp
from ctr_framework.ksconstraints_comp import KSConstraintsComp
from ctr_framework.ksconstraints_min_comp import KSConstraintsMinComp
from ctr_framework.reducedimension_comp import ReducedimensionComp
from ctr_framework.strainvirtual_comp import StrainvirtualComp
'objective'
from ctr_framework.objs_comp import ObjsComp
from ctr_framework.equdply_comp import EqudplyComp
from ctr_framework.reachtargetpts_comp import ReachtargetptsComp
from ctr_framework.targetnorm_comp import TargetnormComp
from ctr_framework.jointvaluereg_comp import JointvalueregComp
from ctr_framework.locnorm_comp import LocnormComp
from ctr_framework.rotnorm_comp import RotnormComp
from ctr_framework.dp_comp import DpComp
from ctr_framework.crosssection_comp import CrosssectionComp
from ctr_framework.signedfun_comp import SignedfunComp

'mesh'
from ctr_framework.mesh_simul import trianglemesh


class CtrsimulGroup():
    '''
    CtrsimulGroup is a OpenMDAO Group object that compassed all the necessary components for solving the 
    CTR design optimization problem. This group includes the CTR kinematics model, kinematics constraints,
    task-specific constraints and objectives, which the user is able to develop their own component and add 
    into the group. CtrsimulGroup simultaneously optimize the number of k inverse kinematics problem and find 
    the optimal design. 
    
    '''
    def initialize(self):
        self.options.declare('num_nodes', default=100, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('k_', default=1, types=int)
        self.options.declare('i')
        self.options.declare('a', default=2, types=int)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('pt')
        self.options.declare('target')
        self.options.declare('rotx_init')
        self.options.declare('roty_init')
        self.options.declare('rotz_init')
        self.options.declare('base')
        self.options.declare('equ_paras')
        self.options.declare('pt_full')
        self.options.declare('viapts_nbr')
        self.options.declare('zeta')
        self.options.declare('rho')
        self.options.declare('lag')
        self.options.declare('meshfile')

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        k_ = self.options['k_']
        i = self.options['i']
        a = self.options['a']
        pt = self.options['pt']
        equ_paras = self.options['equ_paras']
        target = self.options['target']
        tube_nbr = self.options['tube_nbr']
        rotx_init = self.options['rotx_init']
        roty_init = self.options['roty_init']
        rotz_init = self.options['rotz_init']
        pt_full = self.options['pt_full']
        viapts_nbr = self.options['viapts_nbr']
        base = self.options['base']
        zeta = self.options['zeta']
        rho = self.options['rho']
        lag = self.options['lag']
        meshfile = self.options['meshfile']
        mesh  = trianglemesh(num_nodes,k,meshfile)   
        
        
        
        