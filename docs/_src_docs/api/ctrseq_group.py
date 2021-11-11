import numpy as np
from docs._src_docs.api.ctrsimul_group import CtrsimulGroup

class CtrseqGroup(CtrsimulGroup):
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
        
        
        
        