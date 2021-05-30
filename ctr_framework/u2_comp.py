import numpy as np
from openmdao.api import ExplicitComponent
from scipy.linalg import block_diag

class U2Comp(ExplicitComponent):

    def initialize(self):
        self.options.declare('E', default=80.0, types=float)
        self.options.declare('J', default=80.0, types=float)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('kb1')
        self.add_input('kb2')
        self.add_input('kb3')
       
        self.add_input('u1',shape=(num_nodes,k,3,3))
        self.add_input('tube_ends',shape=(num_nodes,k,3))

        # outputs
        self.add_output('u2',shape=(num_nodes,k,3,1))

        ind_kb1 = np.arange(0,3,1)
        indkb = np.arange(0,num_nodes*k*3,3).reshape(-1,1)
        row_indices_kb = (np.tile(ind_kb1,num_nodes*k).reshape(num_nodes*k,len(ind_kb1)) +  indkb).flatten()
        self.declare_partials('u2', 'kb1', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        self.declare_partials('u2', 'kb2', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        self.declare_partials('u2', 'kb3', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        row_indices_u1 = np.outer(np.arange(num_nodes*k*3),np.ones(9)).flatten()
        col_indices_u1 = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2,3,4,5,6,7,8])).flatten()) + (np.arange(0,num_nodes*k*3*3,9).reshape(-1,1))
        self.declare_partials('u2', 'u1',rows=row_indices_u1,cols=col_indices_u1.flatten())
        row_indices_S = np.outer(np.arange(0,num_nodes*k*3),np.ones(3))
        row_indices_s = np.outer(np.arange(num_nodes*k*3),np.ones(3)).flatten()
        col_indices_s = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('u2', 'tube_ends', rows=row_indices_s.flatten(), cols=col_indices_s.flatten())
        


        

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        u1 = inputs['u1']
        tube_ends = inputs['tube_ends']
        u2 = np.zeros((num_nodes,k,3,3))
        tube_ends_kb = np.zeros((num_nodes,k,3))
        tube_ends_kb[:,:,0] = tube_ends[:,:,0] * kb1
        tube_ends_kb[:,:,1] = tube_ends[:,:,1] * kb2
        tube_ends_kb[:,:,2] = tube_ends[:,:,2] * kb3
        
        u2[:,:,:,0] = u1[:,:,:,0] * tube_ends_kb
        u2[:,:,:,1] = u1[:,:,:,1] * tube_ends_kb
        u2[:,:,:,2] = u1[:,:,:,2] * tube_ends_kb

        u2 = np.sum(u2,2)
        outputs['u2'] = np.reshape(u2,(num_nodes,k,3,1))
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        u1 = inputs['u1']
        tube_ends = inputs['tube_ends']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        

        
        # partial
        'u2/kb1'
        tube_ends_kb1 = np.zeros((num_nodes,k,3))
        tube_ends_kb1[:,:,0] = tube_ends[:,:,0]
        temp11 = np.zeros((num_nodes,k,3))
        temp21 = np.zeros((num_nodes,k,3))
        temp31 = np.zeros((num_nodes,k,3))
        temp11 = u1[:,:,:,0] * tube_ends_kb1
        temp21 = u1[:,:,:,1] * tube_ends_kb1
        temp31 = u1[:,:,:,2] * tube_ends_kb1
        temp11= np.sum(temp11,2)
        temp21 = np.sum(temp21,2)
        temp31 = np.sum(temp31,2)
        Pu2_pkb1 = np.zeros((num_nodes*k,3))
        Pu2_pkb1[:, 0] = temp11.flatten()
        Pu2_pkb1[:, 1] = temp21.flatten()
        Pu2_pkb1[:, 2] = temp31.flatten()
        partials['u2','kb1'][:] = Pu2_pkb1.flatten()
        'u2/kb2'
        tube_ends_kb2 = np.zeros((num_nodes,k,3))
        tube_ends_kb2[:,:,1] = tube_ends[:,:,1]
        temp12 = np.zeros((num_nodes,k,3))
        temp22 = np.zeros((num_nodes,k,3))
        temp32 = np.zeros((num_nodes,k,3))
        temp12 = u1[:,:,:,0] * tube_ends_kb2
        temp22 = u1[:,:,:,1] * tube_ends_kb2
        temp32 = u1[:,:,:,2] * tube_ends_kb2
        temp12 = np.sum(temp12,2)
        temp22 = np.sum(temp22,2)
        temp32 = np.sum(temp32,2)
        Pu2_pkb2 = np.zeros((num_nodes*k,3))
        Pu2_pkb2[:, 0] = temp12.flatten()
        Pu2_pkb2[:, 1] = temp22.flatten()
        Pu2_pkb2[:, 2] = temp32.flatten()
        partials['u2','kb2'][:] = Pu2_pkb2.flatten()
        'u2/kb3'
        tube_ends_kb3 = np.zeros((num_nodes,k,3))
        tube_ends_kb3[:,:,2] = tube_ends[:,:,2]
        temp13 = np.zeros((num_nodes,k,3))
        temp23 = np.zeros((num_nodes,k,3))
        temp33 = np.zeros((num_nodes,k,3))
        temp13 = u1[:,:,:,0] * tube_ends_kb3
        temp23 = u1[:,:,:,1] * tube_ends_kb3
        temp33 = u1[:,:,:,2] * tube_ends_kb3
        temp13 = np.sum(temp13,2)
        temp23 = np.sum(temp23,2)
        temp33 = np.sum(temp33,2)
        Pu2_pkb3 = np.zeros((num_nodes*k,3))
        Pu2_pkb3[:, 0] = temp13.flatten()
        Pu2_pkb3[:, 1] = temp23.flatten()
        Pu2_pkb3[:, 2] = temp33.flatten()
        partials['u2','kb3'][:] = Pu2_pkb3.flatten()
        'u2/u1'
        tube_ends_u1 = np.zeros((num_nodes,k,3))
        tube_ends_u1[:,:,0] = tube_ends[:,:,0] * kb1
        tube_ends_u1[:,:,1] = tube_ends[:,:,1] * kb2
        tube_ends_u1[:,:,2] = tube_ends[:,:,2] * kb3
        
        Pu2_pu1 = np.zeros((num_nodes,k,3,9))
        Pu2_pu1[:,:,0,0] = kb1 * tube_ends[:,:,0]
        Pu2_pu1[:,:,0,3] = kb2 * tube_ends[:,:,1]
        Pu2_pu1[:,:,0,6] = kb3 * tube_ends[:,:,2]
        Pu2_pu1[:,:,1,1] = kb1 * tube_ends[:,:,0]
        Pu2_pu1[:,:,1,4] = kb2 * tube_ends[:,:,1]
        Pu2_pu1[:,:,1,7] = kb3 * tube_ends[:,:,2]
        Pu2_pu1[:,:,2,2] = kb1 * tube_ends[:,:,0]
        Pu2_pu1[:,:,2,5] = kb2 * tube_ends[:,:,1]
        Pu2_pu1[:,:,2,8] = kb3 * tube_ends[:,:,2]

        
        partials['u2','u1'][:] = Pu2_pu1.flatten()
        

        'u2/tube_ends'
        tmp = np.zeros((num_nodes,k,3,3))
        tube_ends_kb = np.zeros((num_nodes,k,3))
        tube_ends_kb[:,:,0] = kb1
        tube_ends_kb[:,:,1] = kb2
        tube_ends_kb[:,:,2] = kb3
        
        tmp[:,:,:,0] = u1[:,:,:,0] * tube_ends_kb
        tmp[:,:,:,1] = u1[:,:,:,1] * tube_ends_kb
        tmp[:,:,:,2] = u1[:,:,:,2] * tube_ends_kb
        #tmp = np.sum(tmp,2)

        Pu2_pt = np.zeros((num_nodes,k,9))
        Pu2_pt[:,:,0] = tmp[:,:,0,0]
        Pu2_pt[:,:,1] = tmp[:,:,1,0]
        Pu2_pt[:,:,2] = tmp[:,:,2,0]
        Pu2_pt[:,:,3] = tmp[:,:,0,1]
        Pu2_pt[:,:,4] = tmp[:,:,1,1]
        Pu2_pt[:,:,5] = tmp[:,:,2,1]
        Pu2_pt[:,:,6] = tmp[:,:,0,2]
        Pu2_pt[:,:,7] = tmp[:,:,1,2]
        Pu2_pt[:,:,8] = tmp[:,:,2,2]
        partials['u2','tube_ends'][:] = Pu2_pt.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=3
    k=1
    comp = IndepVarComp()
    comp.add_output('kb1', val=1.654)
    comp.add_output('kb2', val=5.8146)
    comp.add_output('kb3', val=70.3552)
    
    tube_init = np.zeros((n,k,3))
    tube_init[:,:,0] = 1
    tube_init[:5,:,1] = 1
    tube_init[:3,:,2] = 1  
    comp.add_output('tube_ends', val=tube_init)
    comp.add_output('u1', val=np.ones((n,k,3,3)))
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = U2Comp(num_nodes=n,k=k)
    group.add_subsystem('ucomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
