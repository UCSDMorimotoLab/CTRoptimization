import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem

class BbpointsComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=4, types=int)
        self.options.declare('num_nodes', default=10, types=int)

    def setup(self):
        
        tube_nbr = self.options['tube_nbr']
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # input
        self.add_input('R',shape=(num_nodes,k,3,3))
        
        # output
        self.add_output('p_dot',shape=(num_nodes,k,3,1))
        
        # partials
        row_indices = np.outer(np.arange(num_nodes*k*3),np.ones(9)).flatten()
        col_indices = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2,3,4,5,6,7,8])).flatten()) + (np.arange(0,num_nodes*k*3*3,9).reshape(-1,1))
        self.declare_partials('p_dot', 'R',rows = row_indices, cols = col_indices.flatten())
        
    def compute(self,inputs,outputs):
        
        
        R = inputs['R']
        e3 = np.zeros((3,1))
        e3[2,:] = 1 
        outputs['p_dot'] = R @ e3


  
        
    def compute_partials(self, inputs, partials):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        
        'R'
        Ppd_pr = np.zeros((num_nodes,k,3,9))
        Ppd_pr[:,:,0,2] = 1
        Ppd_pr[:,:,1,5] = 1
        Ppd_pr[:,:,2,8] = 1
    
        partials['p_dot','R'][:] = Ppd_pr.flatten()

        


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n=175
    k=1

    comp.add_output('R', val=np.random.random((n,k,3,3)))
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = BbpointsComp(num_nodes=n,k=k)
    group.add_subsystem('bbpoints', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)