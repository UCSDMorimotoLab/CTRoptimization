import numpy as np
import scipy
import os
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None
from ctrsimul_group import CtrsimulGroup


class DualctrsimulGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', default=100, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('k_', default=1, types=int)
        self.options.declare('i')
        self.options.declare('a', default=2, types=int)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('pt')
        self.options.declare('tar_vector')
        self.options.declare('rotx_init')
        self.options.declare('roty_init')
        self.options.declare('rotz_init')
        self.options.declare('base')
        self.options.declare('equ_paras')
        self.options.declare('viapts_nbr')
        self.options.declare('zeta')
        self.options.declare('rho')
        self.options.declare('lag')
        self.options.declare('sp')
        self.options.declare('fp')
        self.options.declare('meshfile')

        
        

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        k_ = self.options['k_']
        i = self.options['i']
        a = self.options['a']
        pt = self.options['pt']
        equ_paras = self.options['equ_paras']
        tar_vector = self.options['tar_vector']
        tube_nbr = self.options['tube_nbr']
        rotx_init = self.options['rotx_init']
        roty_init = self.options['roty_init']
        rotz_init = self.options['rotz_init']
        viapts_nbr = self.options['viapts_nbr']
        base = self.options['base']
        zeta = self.options['zeta']
        rho = self.options['rho']
        lag = self.options['lag']
        sp = self.options['sp']
        fp = self.options['fp']
        meshfile = self.options['meshfile']
        
        # Left-arm
        robot1 = scipy.io.loadmat('leftarm.mat')
        # Right-arm
        robot2 = scipy.io.loadmat('rightarm.mat')
        
        # add subsystem
        'ctr'
        # zeta = np.sum(zeta,axis=0)/3
        # rho = np.sum(rho,axis=0/3)
        # lag = np.sum(lag,axis=0/3)
        #fff1 20,15
        # rho[:,:,:] = rho[:,:,:] * 15
        # rho[:,-1,:] = rho[:,-1,:] * 5
        # zeta[:,:,:] = zeta[:,:,:] * 50
        # zeta[2,:,:] = zeta[2,:,:] * 50
        # heartcase01
        mesh1_adrs = meshfile
        path1_adrs = 'leftpath.mat'
        # jointvalues1_adrs = '/home/fred/Desktop/ctr_optimization/code_opts_seqv2_simulti/results/simul1.mat'
        jointvalues1_adrs = robot1
        ctr1group = CtrsimulGroup(k=k, k_=k_, num_nodes=num_nodes, a=a, i=1, \
                        pt=pt[0,:,:], tar_vector=tar_vector[0,:,:], mesh_adrs = mesh1_adrs,\
                        jointvalues_adrs = jointvalues1_adrs,sp=sp[0,:],fp=fp[0,:],path_adrs = path1_adrs,\
                        viapts_nbr=viapts_nbr, zeta = zeta[0,-1,:],rho=rho[0,:,:], lag=lag[0,:,:],\
                        rotx_init=rotx_init[0],roty_init=roty_init[0],rotz_init=rotz_init[0],base = base[0,:].reshape(3,1),equ_paras = equ_paras)
        
        # heartcase03
        mesh2_adrs = meshfile
        path2_adrs = 'rightpath.mat'
        # jointvalues2_adrs = '/home/fred/Desktop/ctr_optimization/code_opts_seqv2_simulti/results/simul2.mat'
        jointvalues2_adrs=robot2
        ctr2group = CtrsimulGroup(k=k, k_=k_, num_nodes=num_nodes, a=a, i=1, \
                        pt=pt[1,:,:],tar_vector=tar_vector[1,:,:], mesh_adrs = mesh2_adrs,\
                        jointvalues_adrs=jointvalues2_adrs,sp=sp[1,:],fp=fp[1,:],path_adrs=path2_adrs,\
                        viapts_nbr=viapts_nbr, zeta = zeta[1,-1,:], rho=rho[1,:,:], lag=lag[1,:,:],\
                        rotx_init=rotx_init[1],roty_init=roty_init[1],rotz_init=rotz_init[1],base = base[1,:].reshape(3,1),equ_paras = equ_paras)


        self.add_subsystem('Ctr1Group', ctr1group)
        self.add_subsystem('Ctr2Group', ctr2group)
        
        # collision between two arms

        # objectives
        self.connect('Ctr1Group.objs','objs_a1')
        self.connect('Ctr2Group.objs','objs_a2')
        objsmulticomp = ObjsmultiComp(k=k,num_nodes=num_nodes)
        self.add_subsystem('ObjsmultiComp', objsmulticomp, promotes=['*'])
        self.add_objective('objsmulti')


        
