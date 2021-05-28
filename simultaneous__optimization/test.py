import numpy as np
from openmdao.api import ExplicitComponent


class TestComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('sumkm', shape=(num_nodes,k,3,3))
        

        # outputs
        self.add_output('K_s',shape=(num_nodes,k,3,3))


        # partials

        ''' kb '''
        rows_indices = np.arange(num_nodes*k*3*3).flatten()
        cols_indices = np.arange(num_nodes*k*3*3).flatten()
        self.declare_partials('K_s', 'sumkm',rows = rows_indices,cols=cols_indices)
        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        
        sumkm = inputs['sumkm']
        
        epsilon = 1e-10
        self.epsilon = epsilon
        outputs['K_s'] = 1/(sumkm + epsilon)
        # print(outputs['K_s'])
    


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        
        sumkm = inputs['sumkm']
        epsilon = self.epsilon
        
        # Psk_pt = np.zeros((num_nodes,k,9,3))
        
        partials['K_s','sumkm'][:] = (-1/(sumkm+epsilon)**2).flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 175
    k = 1
    comp = IndepVarComp()
    # comp.add_output('tube_section_length', val=0.1 * np.ones((2,3)))
    # comp.add_output('beta', val=0.1 * np.ones((2,3)))
    

    comp.add_output('sumkm', val = np.zeros((n,k,3,3)))
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TestComp(k=k,num_nodes=n)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    