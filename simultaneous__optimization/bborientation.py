import numpy as np
from openmdao.api import ExplicitComponent,Group,Problem

class BborientationComp(ExplicitComponent):
    
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
        self.add_input('uhat',shape=(num_nodes,k,3,3))
        
        
        # output
        self.add_output('R_dot',shape=(num_nodes,k,3,3))
        
        # partials
        
        row_indices = np.outer(np.arange(num_nodes*k*3*3),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(num_nodes*k*3),np.outer(np.ones(3),np.array([0,1,2])).flatten()) \
                                + (np.arange(0,num_nodes*k*3*3,3).reshape(-1,1))
        self.declare_partials('R_dot', 'R',rows = row_indices, cols = col_indices.flatten())
        row_indices_h = np.outer(np.arange(num_nodes*k*3*3),np.ones(9)).flatten()
        col_indices_h = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2,3,4,5,6,7,8])).flatten()) \
                                + (np.arange(0,num_nodes*k*3*3,9).reshape(-1,1))
        self.declare_partials('R_dot','uhat', rows= row_indices_h.flatten(), cols = col_indices_h.flatten())
        
    def compute(self,inputs,outputs):
        
        
        R = inputs['R']
        uhat = inputs['uhat']
        outputs['R_dot'] = R @ uhat
        
    def compute_partials(self, inputs, partials):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        R = inputs['R']
        uhat = inputs['uhat']
        
        'R'
        Prd_pr = np.zeros((num_nodes,k,9,3))
        Prd_pr[:,:,0,0] = uhat[:,:,0,0]
        Prd_pr[:,:,0,1] = uhat[:,:,1,0]
        Prd_pr[:,:,0,2] = uhat[:,:,2,0]
        Prd_pr[:,:,1,0] = uhat[:,:,0,1]
        Prd_pr[:,:,1,1] = uhat[:,:,1,1]
        Prd_pr[:,:,1,2] = uhat[:,:,2,1]
        Prd_pr[:,:,2,0] = uhat[:,:,0,2]
        Prd_pr[:,:,2,1] = uhat[:,:,1,2]
        Prd_pr[:,:,2,2] = uhat[:,:,2,2]

        Prd_pr[:,:,3,0] = uhat[:,:,0,0]
        Prd_pr[:,:,3,1] = uhat[:,:,1,0]
        Prd_pr[:,:,3,2] = uhat[:,:,2,0]
        Prd_pr[:,:,4,0] = uhat[:,:,0,1]
        Prd_pr[:,:,4,1] = uhat[:,:,1,1]
        Prd_pr[:,:,4,2] = uhat[:,:,2,1]
        Prd_pr[:,:,5,0] = uhat[:,:,0,2]
        Prd_pr[:,:,5,1] = uhat[:,:,1,2]
        Prd_pr[:,:,5,2] = uhat[:,:,2,2]

        Prd_pr[:,:,6,0] = uhat[:,:,0,0]
        Prd_pr[:,:,6,1] = uhat[:,:,1,0]
        Prd_pr[:,:,6,2] = uhat[:,:,2,0]
        Prd_pr[:,:,7,0] = uhat[:,:,0,1]
        Prd_pr[:,:,7,1] = uhat[:,:,1,1]
        Prd_pr[:,:,7,2] = uhat[:,:,2,1]
        Prd_pr[:,:,8,0] = uhat[:,:,0,2]
        Prd_pr[:,:,8,1] = uhat[:,:,1,2]
        Prd_pr[:,:,8,2] = uhat[:,:,2,2]

        partials['R_dot','R'][:] = Prd_pr.flatten()

        'uhat'
        Prd_ph = np.zeros((num_nodes,k,9,9))
        Prd_ph[:,:,0,0] = R[:,:,0,0]
        Prd_ph[:,:,0,3] = R[:,:,0,1]
        Prd_ph[:,:,0,6] = R[:,:,0,2]
        Prd_ph[:,:,1,1] = R[:,:,0,0]
        Prd_ph[:,:,1,4] = R[:,:,0,1]
        Prd_ph[:,:,1,7] = R[:,:,0,2]
        Prd_ph[:,:,2,2] = R[:,:,0,0]
        Prd_ph[:,:,2,5] = R[:,:,0,1]
        Prd_ph[:,:,2,8] = R[:,:,0,2]

        Prd_ph[:,:,3,0] = R[:,:,1,0]
        Prd_ph[:,:,3,3] = R[:,:,1,1]
        Prd_ph[:,:,3,6] = R[:,:,1,2]
        Prd_ph[:,:,4,1] = R[:,:,1,0]
        Prd_ph[:,:,4,4] = R[:,:,1,1]
        Prd_ph[:,:,4,7] = R[:,:,1,2]
        Prd_ph[:,:,5,2] = R[:,:,1,0]
        Prd_ph[:,:,5,5] = R[:,:,1,1]
        Prd_ph[:,:,5,8] = R[:,:,1,2]

        Prd_ph[:,:,6,0] = R[:,:,2,0]
        Prd_ph[:,:,6,3] = R[:,:,2,1]
        Prd_ph[:,:,6,6] = R[:,:,2,2]
        Prd_ph[:,:,7,1] = R[:,:,2,0]
        Prd_ph[:,:,7,4] = R[:,:,2,1]
        Prd_ph[:,:,7,7] = R[:,:,2,2]
        Prd_ph[:,:,8,2] = R[:,:,2,0]
        Prd_ph[:,:,8,5] = R[:,:,2,1]
        Prd_ph[:,:,8,8] = R[:,:,2,2]


        partials['R_dot','uhat'][:] = Prd_ph.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n=175
    k=1

    comp.add_output('R', val=np.random.random((n,k,3,3)))
    comp.add_output('uhat', val=np.random.random((n,k,3,3)))
    
    
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = BborientationComp(num_nodes=n,k=k)
    group.add_subsystem('bborientation', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)