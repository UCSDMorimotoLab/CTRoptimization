import numpy as np
import scipy.io
import openmdao.api as om 
from openmdao.api import Problem, Group, ExecComp, IndepVarComp, ScipyOptimizeDriver, pyOptSparseDriver
# from openmdao.api import Problem, Group, ExecComp, IndepVarComp, ScipyOptimizeDriver
from ozone.api import ODEIntegrator
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
from ctr_framework.baseplanar_comp import BaseplanarComp
'constraints'
from ctr_framework.diameter_comp import DiameterComp
from ctr_framework.tubeclearance_comp import TubeclearanceComp
from ctr_framework.bc_comp import BcComp
from ctr_framework.desiredpoints_comp import DesiredpointsComp
from ctr_framework.deployedlength_comp import DeployedlengthComp
from ctr_framework.beta_comp import BetaComp
from ctr_framework.pathpoints_comp import PathpointsComp
from ctr_framework.tubestraight_comp import TubestraightComp
from ctr_framework.tiporientation_comp import TiporientationComp
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
from ctr_framework.mesh import trianglemesh


class CtrseqGroup(om.Group):
    '''
    CtrseqGroup is a OpenMDAO Group object that  all the necessary components for solving the 
    CTR design optimization problem. This group includes the CTR kinematics model, kinematics constraints,
    task-specific constraints and objectives, which the user is able to develop their own component and add 
    into the group. CtrseqGroup aims to obtain a sequence of robot configurations as initial guesses for the next 
    step.
    
    '''
    def initialize(self):
        self.options.declare('num_nodes', default=100, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('i')
        self.options.declare('a', default=2, types=int)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('pt')
        self.options.declare('pt_test')
        self.options.declare('target')
        self.options.declare('zeta')
        self.options.declare('rho')
        self.options.declare('eps_e')
        self.options.declare('eps_r')
        self.options.declare('eps_p')
        self.options.declare('lag')
        self.options.declare('rotx_init')
        self.options.declare('roty_init')
        self.options.declare('rotz_init')
        self.options.declare('base')
        self.options.declare('count')
        self.options.declare('equ_paras')
        self.options.declare('center')
        self.options.declare('pt_full')
        self.options.declare('viapts_nbr')
        self.options.declare('meshfile')
        
        

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        i = self.options['i']
        a = self.options['a']
        equ_paras = self.options['equ_paras']
        pt = self.options['pt']
        pt_test = self.options['pt_test']
        count = self.options['count']
        zeta = self.options['zeta']
        rho = self.options['rho']
        eps_e = self.options['eps_e']
        eps_r = self.options['eps_r']
        eps_p = self.options['eps_p']
        lag = self.options['lag']
        target = self.options['target']
        tube_nbr = self.options['tube_nbr']
        rotx_init = self.options['rotx_init']
        roty_init = self.options['roty_init']
        rotz_init = self.options['rotz_init']
        base = self.options['base']
        center = self.options['center']
        pt_full = self.options['pt_full']
        viapts_nbr = self.options['viapts_nbr']
        meshfile = self.options['meshfile']
        # mesh processing
        mesh  = trianglemesh(num_nodes,k,pt,center,meshfile)  
        p_ = mesh.p
        normals = mesh.normals
        
        
        if i == 0:
            init_guess = scipy.io.loadmat('initial.mat')

        else:
            init_guess = scipy.io.loadmat('seq_'+str(i-1)+'.mat')
            
        
        
        comp = IndepVarComp(num_nodes=num_nodes,k=k)
        comp.add_output('d1', val=init_guess['d1'])
        comp.add_output('d2', val=init_guess['d2'])
        comp.add_output('d3', val=init_guess['d3'])
        comp.add_output('d4', val=init_guess['d4'])
        comp.add_output('d5', val=init_guess['d5'])
        comp.add_output('d6', val=init_guess['d6'])
        comp.add_output('kappa', shape=(1,3), val=init_guess['kappa'])
        comp.add_output('tube_section_length',shape=(1,3),val=init_guess['tube_section_length'])
        comp.add_output('tube_section_straight',shape=(1,3),val=init_guess['tube_section_straight'])
        comp.add_output('alpha', shape=(k,3),val=init_guess['alpha'])
        comp.add_output('beta', shape=(k,3),val=init_guess['beta']+0.01)
        comp.add_output('initial_condition_dpsi', shape=(k,3), val=init_guess['initial_condition_dpsi'])
        comp.add_output('rotx',val=init_guess['rotx'])
        comp.add_output('roty',val=init_guess['roty'])
        comp.add_output('rotz',val=init_guess['rotz'])
        comp.add_output('loc',shape=(3,1),val=init_guess['loc'])
        self.add_subsystem('input_comp', comp, promotes=['*'])


        # add subsystem

        'tube twist'
        stiffness_comp = StiffnessComp()
        self.add_subsystem('stiffness_comp', stiffness_comp, promotes=['*'])
        tube_ends_comp = TubeendsComp(num_nodes=num_nodes,k=k,a=a)
        self.add_subsystem('tube_ends_comp', tube_ends_comp, promotes=['*'])
        interpolationkb_comp =  InterpolationkbComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('interpolationkb_comp', interpolationkb_comp, promotes=['*'])
        tensor_comp = TensorComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('tensor_comp', tensor_comp, promotes=['*'])
        sumkm_comp = SumkmComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('sumkm_comp', sumkm_comp, promotes=['*'])
        invsumk_comp = InvsumkComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('invsumk_comp', invsumk_comp, promotes=['*'])
        k_comp = KComp(num_nodes=num_nodes, k=k)
        self.add_subsystem('k_comp', k_comp, promotes=['*'])
        straightedns_comp = StraightendsComp(num_nodes=num_nodes, k=k,a=a)
        self.add_subsystem('straightends_comp', straightedns_comp, promotes=['*'])
        interpolationkp_comp =  InterpolationkpComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('interpolationkp_comp', interpolationkp_comp, promotes=['*'])
        kappa_comp = KappaComp(num_nodes=num_nodes, k=k)
        self.add_subsystem('kappa_comp', kappa_comp, promotes=['*'])
        kout_comp = KoutComp(num_nodes=num_nodes, k=k)
        self.add_subsystem('kout_comp', kout_comp, promotes=['*'])
        initialpsi_comp = InitialpsiComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('initialpsi_comp', initialpsi_comp, promotes=['*'])
        finaltime_comp = FinaltimeComp()
        self.add_subsystem('final_comp', finaltime_comp, promotes=['*'])

        method_name = 'Lobatto2'
        'ODE 1 : kinematics'
        ode_function1 = CtrFunction(k=1)
        formulation1 = 'time-marching'

        initial_time = 0.
        normalized_times = np.linspace(0., 1, num_nodes)
        
        integrator1 = ODEIntegrator(
            ode_function1, formulation1, method_name,
            initial_time=initial_time, normalized_times=normalized_times
        )

        self.add_subsystem('integrator_group1', integrator1)
        self.connect('final_time', 'integrator_group1.final_time')
        self.connect('K_out', 'integrator_group1.dynamic_parameter:K_out')
        self.connect('initial_condition_psi', 'integrator_group1.initial_condition:psi')
        self.connect('initial_condition_dpsi', 'integrator_group1.initial_condition:dpsi_ds')
        self.connect('integrator_group1.state:dpsi_ds','dpsi_ds')
        self.connect('integrator_group1.state:psi','psi')

        'backbone'
        u1_comp = U1Comp(num_nodes=num_nodes,k=k)
        self.add_subsystem('u1_comp', u1_comp, promotes=['*'])
        u2_comp = U2Comp(num_nodes=num_nodes,k=k)
        self.add_subsystem('u2_comp', u2_comp, promotes=['*'])
        u3_comp = U3Comp(num_nodes=num_nodes,k=k)
        self.add_subsystem('u3_comp', u3_comp, promotes=['*'])
        u_comp = UComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('u_comp', u_comp, promotes=['*'])
        uhat_comp = UhatComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('uhat_comp', uhat_comp, promotes=['*'])
        initR_comp = InitialRComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('initR_comp', initR_comp, promotes=['*'])
        
        'ODE 2: Orientation'
        ode_function2 = BackboneFunction(k=1)
        formulation2 = 'time-marching'

        initial_time = 0.
        normalized_times = np.linspace(0., 1, num_nodes)
        integrator2 = ODEIntegrator(
            ode_function2, formulation2, method_name,
            initial_time=initial_time, normalized_times=normalized_times
        )

        self.add_subsystem('integrator_group2', integrator2)
        self.connect('final_time', 'integrator_group2.final_time')
        self.connect('uhat', 'integrator_group2.dynamic_parameter:uhat')
        self.connect('initial_condition_R', 'integrator_group2.initial_condition:R')

        'ODE 3: Position'
        ode_function3 = BackboneptsFunction(k=1)
        formulation3 = 'time-marching'

        initial_time = 0.
        normalized_times = np.linspace(0., 1, num_nodes)
        initial_conditions = { 'p': np.zeros((k,3,1))
        }
        integrator3 = ODEIntegrator(
            ode_function3, formulation3, method_name,
            initial_time=initial_time, normalized_times=normalized_times,
            initial_conditions=initial_conditions
        )

        self.add_subsystem('integrator_group3', integrator3)
        self.connect('final_time', 'integrator_group3.final_time')
        self.connect('integrator_group2.state:R','integrator_group3.dynamic_parameter:R')
        self.connect('integrator_group3.state:p','p')

        'Transformation'
        baseanglecomp = BaseangleComp(k=k,num_nodes=num_nodes,rotx_init=rotx_init,roty_init=roty_init,rotz_init=rotz_init)
        self.add_subsystem('BaseangleComp', baseanglecomp, promotes=['*'])
        rotpcomp = RotpComp(k=k,num_nodes=num_nodes,base=base)
        self.add_subsystem('RotpComp', rotpcomp, promotes=['*'])



        "Deisgn variables"
        self.add_design_var('d1',lower= 0.2 , upper=3.5)
        self.add_design_var('d2',lower= 0.2, upper=3.5)
        self.add_design_var('d3',lower= 0.2, upper=3.5)
        self.add_design_var('d4',lower= 0.2, upper=3.5)
        self.add_design_var('d5',lower= 0.2, upper=3.5)
        self.add_design_var('d6',lower= 0.2, upper=3.5)
        tube_length_init = 0
        tube_straight_init = 0
        self.add_design_var('tube_section_length',lower=0)
        self.add_design_var('tube_section_straight',lower=0 )
        self.add_design_var('alpha')
        temp = np.outer(np.ones(k) , -init_guess['tube_section_length']+ 2)
        self.add_design_var('beta', upper=-1)
        self.add_design_var('kappa', lower=0)
        self.add_design_var('initial_condition_dpsi')
        self.add_design_var('rotx')
        self.add_design_var('roty')
        self.add_design_var('rotz')
        self.add_design_var('loc')
        locnorm = LocnormComp(k=k,num_nodes=num_nodes)                                
        self.add_subsystem('LocnormComp', locnorm, promotes=['*'])
        rotnorm = RotnormComp(k=k)                                
        self.add_subsystem('rotnormComp', rotnorm, promotes=['*'])


        '''Constraints'''
        bccomp = BcComp(num_nodes=num_nodes,k=k)
        diametercomp = DiameterComp()
        tubeclearancecomp = TubeclearanceComp()
        tubestraightcomp = TubestraightComp()
        baseplanarcomp = BaseplanarComp(num_nodes=num_nodes,k=k,equ_paras=equ_paras)
        # tiporientationcomp = TiporientationComp(k=k,tar_vector=tar_vector)
        deployedlenghtcomp = DeployedlengthComp(k=k)
        betacomp = BetaComp(k=k)

        self.add_subsystem('BetaComp', betacomp, promotes=['*'])
        self.add_subsystem('BcComp', bccomp, promotes=['*'])
        self.add_subsystem('Baseplanarcomp', baseplanarcomp, promotes=['*'])
        self.add_subsystem('DeployedlengthComp', deployedlenghtcomp, promotes=['*'])
        self.add_subsystem('TubestraightComp', tubestraightcomp, promotes=['*'])
        self.add_subsystem('DiameterComp', diametercomp, promotes=['*'])
        self.add_subsystem('TubeclearanceComp', tubeclearancecomp, promotes=['*'])
        # self.add_subsystem('TiporientationComp', tiporientationcomp, promotes=['*'])

        #strain
        num_t = 2 
        ksconstraintscomp = KSConstraintsComp(
        in_name='strain_virtual',
        out_name='strain_max',
        shape=(num_nodes,k,num_t,tube_nbr),
        axis=0,
        rho=100.,
        )
        ksconstraintsmincomp = KSConstraintsMinComp(
        in_name='strain_virtual',
        out_name='strain_min',
        shape=(num_nodes,k,num_t,tube_nbr),
        axis=0,
        rho=100.,
        )
        kappaeqcomp = KappaeqComp(num_nodes=num_nodes,k=k)
        gammacomp = GammaComp(num_nodes=num_nodes,k=k)
        chicomp = ChiComp(num_nodes=num_nodes,k=k,num_t = num_t)
        straincomp = StrainComp(num_nodes = num_nodes,k=k,num_t= num_t)
        strainvirtualcomp = StrainvirtualComp(num_nodes = num_nodes,k=k,num_t= num_t)
        self.add_subsystem('KappaeqComp', kappaeqcomp, promotes=['*'])
        self.add_subsystem('GammaComp', gammacomp, promotes=['*'])
        self.add_subsystem('ChiComp', chicomp, promotes=['*'])
        self.add_subsystem('StrainComp', straincomp, promotes=['*'])
        self.add_subsystem('StrainvirtualComp', strainvirtualcomp, promotes=['*'])
        self.add_subsystem('KsconstraintsComp', ksconstraintscomp, promotes=['*'])
        self.add_subsystem('KsconstraintsminComp', ksconstraintsmincomp, promotes=['*'])


        self.add_constraint('torsionconstraint', equals=0.)
        # self.add_constraint('tiporientation', equals=0)
        self.add_constraint('locnorm', upper=2)
        # self.add_constraint('baseconstraints', lower=0)
        self.add_constraint('deployedlength12constraint', lower=1)
        self.add_constraint('deployedlength23constraint', lower=1)
        self.add_constraint('beta12constraint', upper=-1)
        self.add_constraint('beta23constraint', upper=-1)
        d_c = np.zeros((1,3)) + 0.1
        self.add_constraint('diameterconstraint',lower= d_c)
        self.add_constraint('tubeclearanceconstraint',lower= 0.1,upper=0.16)
        self.add_constraint('tubestraightconstraint',lower= 0)
        self.add_constraint('strain_max',upper=0.08)
        self.add_constraint('strain_min',lower = -0.08)
        
        
        '''objectives'''
        desiredpointscomp = DesiredpointsComp(num_nodes=num_nodes,k=k)
        self.add_subsystem('Desiredpointscomp', desiredpointscomp, promotes=['*'])
        reachtargetptscomp = ReachtargetptsComp(k=k,targets = pt)
        self.add_subsystem('reachtargetptsComp', reachtargetptscomp, promotes=['*'])
        targetnormcomp = TargetnormComp(k=k)
        self.add_subsystem('Targetnormcomp', targetnormcomp, promotes=['*'])
        dpcomp = DpComp(k=k,num_nodes=num_nodes,p_=p_)
        self.add_subsystem('DpComp', dpcomp, promotes=['*'])
        crosssectioncomp = CrosssectionComp(k=k,num_nodes=num_nodes)
        self.add_subsystem('CrosssectionComp', crosssectioncomp, promotes=['*'])
        signedfuncomp = SignedfunComp(k=k,num_nodes=num_nodes,normals=normals)
        self.add_subsystem('SignedfunComp', signedfuncomp, promotes=['*'])
        equdply = EqudplyComp(k=k,num_nodes=num_nodes)
        self.add_subsystem('EqudplyComp', equdply, promotes=['*'])

        # objective function
        dl0 = init_guess['tube_section_length'] + init_guess['beta']
        norm1 = np.linalg.norm(pt_full[0,:]-pt_full[-1,:],ord=1.125)
        norm2 = (dl0[:,0] - dl0[:,1])**2 + (dl0[:,1] -  dl0[:,2])**2
        norm3 = np.linalg.norm(pt_full[0,:]-pt_full[-1,:])/viapts_nbr
        norm4 = 2
        norm5 = 2*np.pi 
        objscomp = ObjsComp(k=k,num_nodes=num_nodes,
                            zeta=zeta,
                                rho=rho,
                                    eps_r=eps_r,
                                        eps_p=eps_p,
                                            lag=lag,
                                                norm1 = norm1,
                                                    norm2 = norm2,
                                                        norm3 = norm3,
                                                            norm4 = norm4,
                                                                norm5 = norm5,
                                                                    eps_e = eps_e,)                                    
        self.add_subsystem('ObjsComp', objscomp, promotes=['*'])
        self.add_objective('objs')
        