import numpy as np
from openmdao.api import ExplicitComponent
from scipy.linalg import block_diag


class KoutComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=40, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('K_kp',shape=(num_nodes,k,3,3))
        self.add_input('K_tube',shape=(num_nodes,k,3,3))
        
        # outputs
        self.add_output('K_out',shape=(num_nodes,k,3,3))


        # partials
        row_indices_K=np.arange(num_nodes*k*3*3).flatten()
        col_indices_K = np.arange(num_nodes*k*3*3).flatten()
        self.declare_partials('K_out', 'K_tube',rows= row_indices_K, cols=col_indices_K)
        self.declare_partials('K_out', 'K_kp',rows= row_indices_K, cols=col_indices_K)
        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        K_tube = inputs['K_tube']
        K_kp = inputs['K_kp']
        
        K_out = K_tube * K_kp    
        outputs['K_out'] = K_out
        

    def compute_partials(self,inputs,partials):
         num_nodes = self.options['num_nodes']
         k = self.options['k']
         tube_nbr = self.options['tube_nbr']
         K_kp = inputs['K_kp']
         K_tube = inputs['K_tube']
         
         """partials Jacobian of partial derivatives."""
        
        
         partials['K_out','K_kp'] = K_tube.flatten()
         partials['K_out','K_tube'] = K_kp.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    
    comp.add_output('K_tube',val = np.ones((40,3,3,3)))
    comp.add_output('K_kp', val=np.random.random((40,3,3,3)))
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = KoutComp()
    group.add_subsystem('kcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    