import numpy as np
from openmdao.api import ExplicitComponent


class KappaComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('straight_ends', shape=(num_nodes,k,3))
        

        # outputs
        self.add_output('K_kp',shape=(num_nodes,k,3,3))
        # partials

        row_indices_S = np.outer(np.arange(0,num_nodes*k*3*3),np.ones(3))
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2])).flatten()) \
                            + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('K_kp', 'straight_ends', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        straight_ends = inputs['straight_ends']
        

        K_kp = np.zeros((num_nodes,k,3,3))
        K_kp[:,:,0,0] = straight_ends[:,:,0] * straight_ends[:,:,0]
        K_kp[:,:,0,1] = straight_ends[:,:,0] * straight_ends[:,:,1]
        K_kp[:,:,0,2] = straight_ends[:,:,0] * straight_ends[:,:,2]
        K_kp[:,:,1,0] = straight_ends[:,:,1] * straight_ends[:,:,0]
        K_kp[:,:,1,1] = straight_ends[:,:,1] * straight_ends[:,:,1]
        K_kp[:,:,1,2] = straight_ends[:,:,1] * straight_ends[:,:,2]
        K_kp[:,:,2,0] = straight_ends[:,:,2] * straight_ends[:,:,0]
        K_kp[:,:,2,1] = straight_ends[:,:,2] * straight_ends[:,:,1]
        K_kp[:,:,2,2] = straight_ends[:,:,2] * straight_ends[:,:,2]

        outputs['K_kp'] = K_kp
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        straight_ends = inputs['straight_ends']


        
        Pkkp_ps = np.zeros((num_nodes,k,9,3))
        Pkkp_ps[:,:,0,0] = 2 * straight_ends[:,:,0]
        Pkkp_ps[:,:,1,0] = straight_ends[:,:,1]
        Pkkp_ps[:,:,2,0] = straight_ends[:,:,2]
        Pkkp_ps[:,:,3,0] = straight_ends[:,:,1]
        Pkkp_ps[:,:,4,0] = 0
        Pkkp_ps[:,:,5,0] = 0
        Pkkp_ps[:,:,6,0] = straight_ends[:,:,2]
        Pkkp_ps[:,:,7,0] = 0
        Pkkp_ps[:,:,8,0] = 0

        Pkkp_ps[:,:,0,1] = 0
        Pkkp_ps[:,:,1,1] = straight_ends[:,:,0]
        Pkkp_ps[:,:,2,1] = 0
        Pkkp_ps[:,:,3,1] = straight_ends[:,:,0]
        Pkkp_ps[:,:,4,1] = 2 * straight_ends[:,:,1]
        Pkkp_ps[:,:,5,1] = straight_ends[:,:,2]
        Pkkp_ps[:,:,6,1] = 0
        Pkkp_ps[:,:,7,1] = straight_ends[:,:,2]
        Pkkp_ps[:,:,8,1] = 0

        Pkkp_ps[:,:,0,2] = 0
        Pkkp_ps[:,:,1,2] = 0
        Pkkp_ps[:,:,2,2] = straight_ends[:,:,0]
        Pkkp_ps[:,:,3,2] = 0
        Pkkp_ps[:,:,4,2] = 0
        Pkkp_ps[:,:,5,2] = straight_ends[:,:,1]
        Pkkp_ps[:,:,6,2] = straight_ends[:,:,0]
        Pkkp_ps[:,:,7,2] = straight_ends[:,:,1]
        Pkkp_ps[:,:,8,2] = 2 * straight_ends[:,:,2]

        partials['K_kp','straight_ends'][:] =  Pkkp_ps.flatten()



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 175
    k = 1
    comp = IndepVarComp()
   
    comp.add_output('straight_ends', val=np.random.random((n,k,3))*10)
    
    

    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = KappaComp(num_nodes=n,k=k)
    group.add_subsystem('kappascomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    