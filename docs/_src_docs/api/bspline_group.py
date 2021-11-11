import numpy as np
import openmdao.api as om 
from openmdao.api import Problem, Group, ExecComp, IndepVarComp
from ozone.api import ODEIntegrator
from ctr_framework.startpoint_comp import StartpointComp
from ctr_framework.finalpoint_comp import FinalpointComp
from ctr_framework.mesh_path import trianglemesh
from ctr_framework.initialize import initialize_bspline
from ctr_framework.bspline_3d_comp import BsplineComp, get_bspline_mtx
from ctr_framework.pt_comp import PtComp
from ctr_framework.signedpt_comp import SignedptComp
from ctr_framework.ptequdistant1_comp import Ptequdistant1Comp
from ctr_framework.ptequdistant2_comp import Ptequdistant2Comp
from ctr_framework.pathobjective_comp import PathobjectiveComp


class BsplineGroup(om.Group):
    '''
    BsplineGroup is a OpenMDAO Group object that includes all the components forming the optimization problem for finding a collision
    free path for the robot to follow. BsplineGroup uses the 3D B-spline function and a continuous function of anatomical constraints
    to ensure the B-spline curve is inside the anatomy. 
    
    '''
    def initialize(self):
        self.options.declare('filename')
        self.options.declare('r2')
        self.options.declare('r1')
        self.options.declare('sp')
        self.options.declare('fp')
        self.options.declare('num_cp', default=25, types=int)
        self.options.declare('num_pt', default=100, types=int)
        
        

    def setup(self):
        filename = self.options['filename']
        r2 = self.options['r2']
        r1 = self.options['r1']
        sp = self.options['sp']
        fp = self.options['fp']
        num_cp = self.options['num_cp']
        num_pt = self.options['num_pt']
        
        # mesh processing
        mesh  = trianglemesh(filename)
        p_ = mesh.p
        normals = mesh.normals
        
        comp = IndepVarComp(num_cp=num_cp,num_pt=num_pt)
        c_points,p_points = initialize_bspline(sp,fp,num_cp,num_pt)
        comp.add_output('cp', val=c_points)
        
        self.add_subsystem('input_comp', comp, promotes=['*'])
        jac = get_bspline_mtx(num_cp,num_pt)
        bspline_comp =  BsplineComp(
        num_cp=num_cp,
        num_pt=num_pt,
        jac=jac,
        in_name='cp',
        out_name='pt',
        )

        self.add_subsystem('Bspline_comp', bspline_comp, promotes=['*'])
        startpointcomp = StartpointComp(num_cp=num_cp)
        Finalpointcomp = FinalpointComp(num_cp=num_cp)
        self.add_subsystem('Startpointcomp', startpointcomp, promotes=['*'])
        self.add_subsystem('Finalpointcomp', Finalpointcomp, promotes=['*'])
        pt_comp = PtComp(num_pt=num_pt,p_=p_,normals=normals)
        self.add_subsystem('Pt_comp',pt_comp,promotes=['*'])
        signedpt_comp = SignedptComp(num_pt=num_pt,p_=p_,normals=normals)
        self.add_subsystem('Signedpt_comp',signedpt_comp,promotes=['*'])
        ptequdistant1_comp = Ptequdistant1Comp(num_pt=num_pt)
        self.add_subsystem('ptequdistant1_comp', ptequdistant1_comp, promotes=['*'])
        ptequdistant2_comp = Ptequdistant2Comp(pt_=p_points,num_pt=num_pt)
        self.add_subsystem('ptequdistant2_comp',ptequdistant2_comp,promotes=['*'])
        norm1 = np.linalg.norm(sp-fp,ord=1.125)
        pathobjective_comp = PathobjectiveComp(r2=r2,r1=r1/norm1)
        self.add_subsystem('pathobjective_comp',pathobjective_comp,promotes=['*'])

        
        # Design variable
        self.add_design_var('cp')

        # Constraints
        self.add_constraint('startpoint_constraint',equals=sp)
        self.add_constraint('finalpoint_constraint',equals=fp)
        
        
        # Objectives
        self.add_objective('path_objective')