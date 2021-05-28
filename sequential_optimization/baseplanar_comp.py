import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class BaseplanarComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)
        self.options.declare('equ_paras')
        

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        

        #Inputs
        self.add_input('rot_p', shape=(num_nodes,k,3,1))

        # outputs
        self.add_output('baseconstraints',shape=(k))
        
        
        col_indices = np.arange(k*3).flatten()
        row_indices = np.outer(np.arange(k),np.ones(3)).flatten()        
        self.declare_partials('baseconstraints', 'rot_p',rows = row_indices, cols = col_indices)

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        rot_p = inputs['rot_p']
        rot_p = np.reshape(rot_p,(num_nodes,k,3))
        equ_paras = self.options['equ_paras']
            
        outputs['baseconstraints'] = equ_paras[0] * rot_p[0,:,0] + equ_paras[1] * rot_p[0,:,1] + \
             equ_paras[2] * rot_p[0,:,2] - equ_paras[3]


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        rot_p = inputs['rot_p']
        equ_paras = self.options['equ_paras']
        
        '''Computing Partials'''
        pb_pp = np.zeros((k,3))
        pb_pp[:,0] = equ_paras[0]
        pb_pp[:,1] = equ_paras[1]
        pb_pp[:,2] = equ_paras[2]

        partials['baseconstraints','rot_p'][:]= pb_pp.flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=3
    k=2
    comp = IndepVarComp()
    comp.add_output('rot_p', val=np.random.random((n,k,3,1)))
    comp.add_output('tube_ends_tip', val=([1.3,3,3],[2.1,3.1,3.1]))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = BaseplanarComp(num_nodes=n,k=k,equ_paras=np.array([3,4,5,6]))
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    # prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
