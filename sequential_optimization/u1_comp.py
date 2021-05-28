import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class U1Comp(ExplicitComponent):

    def initialize(self):
        self.options.declare('E', default=80.0, types=float)
        self.options.declare('J', default=80.0, types=float)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    '''This class is defining the sin() tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('psi', shape=(num_nodes,k,3))
        self.add_input('dpsi_ds', shape=(num_nodes,k,3))
        self.add_input('straight_ends',shape=(num_nodes,k,3))

        # outputs
        self.add_output('u1',shape=(num_nodes,k,3,3))
        
        row_indices_S = np.outer(np.arange(0,num_nodes*k*3*3),np.ones(3))
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2])).flatten())\
                            + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('u1', 'psi', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
        self.declare_partials('u1', 'dpsi_ds', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
        self.declare_partials('u1', 'straight_ends',rows=row_indices_S.flatten(), cols=col_indices_S.flatten())

       
        

        

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        dpsi_ds = inputs['dpsi_ds']
        psi = inputs['psi']
        straight_ends = inputs['straight_ends']
        

        
        # compute tensor
        u = np.zeros((num_nodes,k,3,1))
        R = np.zeros((num_nodes,k,3,3,3))
        psi1 = np.zeros((num_nodes,k,3))
        psi1[:,:,0] = psi[:,:,0]
        psi1[:,:,1] = psi[:,:,0]
        psi1[:,:,2] = psi[:,:,0]
        R[:,:,:,0,0] = np.cos(psi-psi1)
        R[:,:,:,0,1] = -np.sin(psi-psi1)
        R[:,:,:,1,0] = np.sin(psi-psi1)
        R[:,:,:,1,1] = np.cos(psi-psi1)
        R[:,:,:,2,2] = np.ones((num_nodes,k,3)) 
        
        kappa = np.zeros((num_nodes,k,3,3,1))
        kappa[:,:,0,0,:] = np.reshape(straight_ends[:,:,0],(num_nodes,k,1))
        kappa[:,:,1,0,:] = np.reshape(straight_ends[:,:,1],(num_nodes,k,1))
        kappa[:,:,2,0,:] = np.reshape(straight_ends[:,:,2],(num_nodes,k,1))
        
        u = R @ kappa
        dpsi_ds[:,:,0] =  dpsi_ds[:,:,0] - dpsi_ds[:,:,0]
        dpsi_ds[:,:,1] =  dpsi_ds[:,:,1] - dpsi_ds[:,:,0]
        dpsi_ds[:,:,2] =  dpsi_ds[:,:,2] - dpsi_ds[:,:,0]

        u[:,:,:,2,:] = u[:,:,:,2,:] - np.reshape(dpsi_ds,(num_nodes,k,3,1))
        outputs['u1'] = np.reshape(u,(num_nodes,k,3,3))


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        psi = inputs['psi']
        straight_ends = inputs['straight_ends']
        
        psi1 = np.zeros((num_nodes,k,3))
        psi1[:,:,0] = psi[:,:,0]
        psi1[:,:,1] = psi[:,:,0]
        psi1[:,:,2] = psi[:,:,0]
        kappa = np.zeros((num_nodes,k,3,3))
        kappa[:,:,0,0] = straight_ends[:,:,0]
        kappa[:,:,1,0] = straight_ends[:,:,1]
        kappa[:,:,2,0] = straight_ends[:,:,2]
        R = np.zeros((num_nodes,k,3,3,3))
        R[:,:,:,0,0] = np.cos(psi-psi1)
        R[:,:,:,0,1] = -np.sin(psi-psi1)
        R[:,:,:,1,0] = np.sin(psi-psi1)
        R[:,:,:,1,1] = np.cos(psi-psi1)
        R[:,:,:,2,2] = np.ones((num_nodes,k,3))
        'dpsi'
        Pu1_pds = np.zeros((num_nodes,k,9,3))

        Pu1_pds[:,:, 5,1] = -1
        Pu1_pds[:,:, 8,2] = -1
        partials['u1','dpsi_ds'] = Pu1_pds.flatten()

        'psi'
        Pu1_ppsi = np.zeros((num_nodes,k,9,3))
        Pu1_ppsi[:,:, 3,0] =  np.sin(psi[:,:,1]-psi1[:,:,0]) * kappa[:,:,1,0]
        Pu1_ppsi[:,:, 4,0] = -np.cos(psi[:,:,1]-psi1[:,:,0]) * kappa[:,:,1,0]
        Pu1_ppsi[:,:, 7,0] = -np.cos(psi[:,:,2]-psi1[:,:,0]) * kappa[:,:,2,0]
        Pu1_ppsi[:,:, 6,0] =  np.sin(psi[:,:,2]-psi1[:,:,0]) * kappa[:,:,2,0]
        
        Pu1_ppsi[:,:, 4,1] = np.cos(psi[:,:,1]-psi1[:,:,0]) * kappa[:,:,1,0]
        Pu1_ppsi[:,:, 3,1] = -np.sin(psi[:,:,1]-psi1[:,:,0]) * kappa[:,:,1,0]
        
        Pu1_ppsi[:,:, 7,2] = np.cos(psi[:,:,2]-psi1[:,:,0]) * kappa[:,:,2,0]
        Pu1_ppsi[:,:, 6,2] = -np.sin(psi[:,:,2]-psi1[:,:,0]) * kappa[:,:,2,0]
        partials['u1','psi'] = Pu1_ppsi.flatten()
        'straight_ends'
        Pu1_ps = np.zeros((num_nodes,k,9,3))
        Pu1_ps[:,:, 0,0] = 1
        Pu1_ps[:,:, 3,1] = np.cos(psi[:,:,1]-psi1[:,:,0])
        Pu1_ps[:,:, 4,1] = np.sin(psi[:,:,1]-psi1[:,:,0])
        Pu1_ps[:,:, 7,2] = np.sin(psi[:,:,2]-psi1[:,:,0])
        Pu1_ps[:,:, 6,2] = np.cos(psi[:,:,2]-psi1[:,:,0])


        partials['u1','straight_ends'] = Pu1_ps.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=3
    k=1
    comp = IndepVarComp()
    comp.add_output('psi', val=np.random.random((n,k,3)))
    comp.add_output('dpsi_ds', val=np.random.random((n,k,3)))
    comp.add_output('straight_ends', val=np.random.random((n,k,3))) 

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = U1Comp(num_nodes=n,k=k)
    group.add_subsystem('ucomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
