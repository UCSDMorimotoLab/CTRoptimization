import numpy as np




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
        
        
        
        