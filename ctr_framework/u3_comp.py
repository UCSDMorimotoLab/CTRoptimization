import numpy as np
from openmdao.api import ExplicitComponent
from scipy.linalg import block_diag

class U3Comp(ExplicitComponent):

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
        self.add_input('kt1')
        self.add_input('kt2')
        self.add_input('kt3')
        self.add_input('tube_ends',shape=(num_nodes,k,3))

        # outputs
        self.add_output('u3',shape=(num_nodes,k,3,3))


        ind_skb1 = np.arange(0,9,1)
        indskb = np.arange(0,num_nodes*k*9,9).reshape(-1,1)
        row_indices_skb = (np.tile(ind_skb1,num_nodes*k).reshape(num_nodes*k,len(ind_skb1)) +  indskb).flatten()
        self.declare_partials('u3', 'kb1', rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kb2', rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kb3', rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kt1',rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kt2',rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        self.declare_partials('u3', 'kt3',rows=row_indices_skb, cols=np.zeros(len(row_indices_skb)).flatten())
        row_indices_st = np.outer(np.arange(0,num_nodes*k*3*3),np.ones(3))
        col_indices_st = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('u3', 'tube_ends', rows=row_indices_st.flatten(), cols=col_indices_st.flatten())
        

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kt1 = inputs['kt1']
        kt2 = inputs['kt2']
        kt3 = inputs['kt3']
        tube_ends = inputs['tube_ends']
        # print(tube_ends)
        u2 = np.zeros((num_nodes,k,3,3))
        tube_ends_kb = np.zeros((num_nodes,k,3))
        tube_ends_kb[:,:,0] = tube_ends[:,:,0] * kb1
        tube_ends_kb[:,:,1] = tube_ends[:,:,1] * kb2
        tube_ends_kb[:,:,2] = tube_ends[:,:,2] * kb3
        
        
        K = np.zeros((num_nodes,k,3,3))
        tube_ends_kt = np.zeros((num_nodes,k,3))
        tube_ends_kt[:,:,0] = tube_ends[:,:,0] * kt1
        tube_ends_kt[:,:,1] = tube_ends[:,:,1] * kt2
        tube_ends_kt[:,:,2] = tube_ends[:,:,2] * kt3
        K[:,:,0,0] =  tube_ends_kb[:,:,0] + tube_ends_kb[:,:,1] + tube_ends_kb[:,:,2] + 1e-10
        K[:,:,1,1] =  tube_ends_kb[:,:,0] + tube_ends_kb[:,:,1] + tube_ends_kb[:,:,2] + 1e-10
        K[:,:,2,2] =  tube_ends_kt[:,:,0] + tube_ends_kt[:,:,1] + tube_ends_kt[:,:,2] + 1e-10

        outputs['u3'] = K 


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_ends = inputs['tube_ends']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kt1 = inputs['kt1']
        kt2 = inputs['kt2']
        kt3 = inputs['kt3']

        
        # partial

        'sk/kb1'
        Psk_pkb1 = np.zeros((num_nodes*k,9))
        Psk_pkb1[:, 0] = tube_ends[:,:,0].flatten()
        Psk_pkb1[:, 4] = tube_ends[:,:,0].flatten()
        partials['u3','kb1'][:] = Psk_pkb1.flatten()
        'sk/kb2'
        Psk_pkb2 = np.zeros((num_nodes*k,9))
        Psk_pkb2[:, 0] = tube_ends[:,:,1].flatten()
        Psk_pkb2[:, 4] = tube_ends[:,:,1].flatten()
        partials['u3','kb2'][:] = Psk_pkb2.flatten()
        'sk/kb3'
        Psk_pkb3 = np.zeros((num_nodes*k,9))
        Psk_pkb3[:, 0] = tube_ends[:,:,2].flatten()
        Psk_pkb3[:, 4] = tube_ends[:,:,2].flatten()
        partials['u3','kb3'][:] = Psk_pkb3.flatten()
        'sk/kt1'
        Psk_pkt1 = np.zeros((num_nodes,k,9))
        Psk_pkt1[:, :,8] = tube_ends[:,:,0]
        partials['u3','kt1'][:] = Psk_pkt1.flatten()
        'sk/kt2'
        Psk_pkt2 = np.zeros((num_nodes,k,9))
        Psk_pkt2[:, :, 8] = tube_ends[:,:,1]
        partials['u3','kt2'][:] = Psk_pkt2.flatten()
        'sk/kt3'
        Psk_pkt3 = np.zeros((num_nodes,k,9))
        Psk_pkt3[:, :,8] = tube_ends[:,:,2]
        partials['u3','kt3'][:] = Psk_pkt3.flatten()
        'sk/tube_ends'
        Psk_pt = np.zeros((num_nodes,k,9,3))
        Psk_pt[:,:, 0,0] = kb1
        Psk_pt[:,:, 4,0] = kb1
        Psk_pt[:,:, 8,0] = kt1

        Psk_pt[:,:, 0,1] = kb2
        Psk_pt[:,:, 4,1] = kb2
        Psk_pt[:,:, 8,1] = kt2

        Psk_pt[:,:, 0,2] = kb3
        Psk_pt[:,:, 4,2] = kb3
        Psk_pt[:,:, 8,2] = kt3

        partials['u3','tube_ends'][:] = Psk_pt.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=50
    k=10
    comp = IndepVarComp()
    comp.add_output('kb1', val=1.654)
    comp.add_output('kb2', val=5.8146)
    comp.add_output('kb3', val=70.3552)
    comp.add_output('kt1', val=1.2405)
    comp.add_output('kt2', val=4.3609)
    comp.add_output('kt3', val=140.7105)
    tube_init = np.zeros((n,k,3))
    tube_init[:,:,0] = 1
    tube_init[:5,:,1] = 1
    tube_init[:3,:,2] = 1  
    comp.add_output('tube_ends', val=tube_init)
        
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = U3Comp(num_nodes=n,k=k)
    group.add_subsystem('ucomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    # prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
