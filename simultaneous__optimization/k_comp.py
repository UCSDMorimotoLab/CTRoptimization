import numpy as np
from openmdao.api import ExplicitComponent
from scipy.linalg import block_diag


class KComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('E', default=80.0, types=float)
        self.options.declare('J', default=80.0, types=float)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=1, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('tube_ends',shape=(num_nodes,k,3))
        self.add_input('K_s',shape=(num_nodes,k,3,3))
        
        # outputs
        self.add_output('K_tube',shape=(num_nodes,k,3,3))

        # partials
        row_indices_K=np.arange(num_nodes*k*3*3).flatten()
        col_indices_K = np.arange(num_nodes*k*3*3).flatten()
        self.declare_partials('K_tube', 'K_s',rows= row_indices_K, cols=col_indices_K)
        row_indices_S = np.outer(np.arange(0,num_nodes*k*3*3),np.ones(3))
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('K_tube', 'tube_ends', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
        
        
        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        K = inputs['K_s']
        tube_ends = inputs['tube_ends']
        K_tube = np.zeros((num_nodes,k,3,3))
        K_tube[:,:,0,0] = K[:,:,0,0] * tube_ends[:,:,0]
        K_tube[:,:,0,1] = K[:,:,0,1] * tube_ends[:,:,0] * tube_ends[:,:,1]
        K_tube[:,:,0,2] = K[:,:,0,2] * tube_ends[:,:,0] * tube_ends[:,:,2]    
        K_tube[:,:,1,0] = K[:,:,1,0] * tube_ends[:,:,1] * tube_ends[:,:,0]
        K_tube[:,:,1,1] = K[:,:,1,1] * tube_ends[:,:,1] 
        K_tube[:,:,1,2] = K[:,:,1,2] * tube_ends[:,:,1] * tube_ends[:,:,2]
        K_tube[:,:,2,0] = K[:,:,2,0] * tube_ends[:,:,2] * tube_ends[:,:,0]
        K_tube[:,:,2,1] = K[:,:,2,1] * tube_ends[:,:,2] * tube_ends[:,:,1]
        K_tube[:,:,2,2] = K[:,:,2,2] * tube_ends[:,:,2]
        
        
        
        outputs['K_tube'] = K_tube
        
        


    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tube_nbr = self.options['tube_nbr']
        tube_ends = inputs['tube_ends']
        K = inputs['K_s']
        
        """partials Jacobian of partial derivatives."""
    
        # K
        Pk_pk = np.zeros((num_nodes,k,9))
        Pk_pk[:,:,0] = tube_ends[:,:,0]
        Pk_pk[:,:,1] = tube_ends[:,:,0] * tube_ends[:,:,1]
        Pk_pk[:,:,2] = tube_ends[:,:,0] * tube_ends[:,:,2]    
        Pk_pk[:,:,3] = tube_ends[:,:,1] * tube_ends[:,:,0]
        Pk_pk[:,:,4] = tube_ends[:,:,1]
        Pk_pk[:,:,5] = tube_ends[:,:,1] * tube_ends[:,:,2]
        Pk_pk[:,:,6] = tube_ends[:,:,2] * tube_ends[:,:,0]
        Pk_pk[:,:,7] = tube_ends[:,:,2] * tube_ends[:,:,1]
        Pk_pk[:,:,8] = tube_ends[:,:,2]
    

        Pk_pt = np.zeros((num_nodes,k,9,3))
        Pk_pt[:,:, 0,0] = K[:,:,0,0]
        Pk_pt[:,:, 1,0] = K[:,:,0,1] * tube_ends[:,:,1]
        Pk_pt[:,:, 2,0] = K[:,:,0,2] * tube_ends[:,:,2]
        Pk_pt[:,:, 3,0] = K[:,:,1,0] * tube_ends[:,:,1]
        Pk_pt[:,:, 6,0] = K[:,:,2,0] * tube_ends[:,:,2]


        Pk_pt[:,:, 1,1] = K[:,:,0,1] * tube_ends[:,:,0]
        Pk_pt[:,:, 3,1] = K[:,:,1,0] * tube_ends[:,:,0]
        Pk_pt[:,:, 4,1] = K[:,:,1,1] 
        Pk_pt[:,:, 5,1] = K[:,:,1,2] * tube_ends[:,:,2]
        Pk_pt[:,:, 7,1] = K[:,:,2,1] * tube_ends[:,:,2]

        Pk_pt[:,:, 2,2] = K[:,:,0,2] * tube_ends[:,:,0]
        Pk_pt[:,:, 5,2] = K[:,:,1,2] * tube_ends[:,:,1]
        Pk_pt[:,:, 6,2] = K[:,:,2,0] * tube_ends[:,:,0]
        Pk_pt[:,:, 7,2] = K[:,:,2,1] * tube_ends[:,:,1]
        Pk_pt[:,:, 8,2] = K[:,:,2,2]
    
    
        partials['K_tube','K_s'] = Pk_pk.flatten()
        partials['K_tube','tube_ends'] = Pk_pt.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n=175
    k=1
    comp.add_output('tube_ends',val = np.random.random((n,k,3))*10)
    comp.add_output('K_s', val=np.random.random((n,k,3,3)))
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = KComp(num_nodes=n,k=k)
    group.add_subsystem('kcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    