import numpy as np
from openmdao.api import ExplicitComponent


class TensorComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('E', default=80.0, types=float)
        self.options.declare('J', default=80.0, types=float)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=40, types=int)
        
        

    '''This class is defining K tensor'''
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


        # outputs
        self.add_output('K',shape=(num_nodes,k,3,3))


        # partials
        ''' kb '''
        ind_kb1 = np.arange(0,9,1)
        indkb = np.arange(0,num_nodes*k*9,9).reshape(-1,1)
        row_indices_kb = (np.tile(ind_kb1,num_nodes*k).reshape(num_nodes*k,len(ind_kb1)) +  indkb).flatten()
        self.declare_partials('K', 'kb1', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        self.declare_partials('K', 'kb2', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        self.declare_partials('K', 'kb3', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())

        ''' kt '''
        ind_kt1 = np.array([0,1,2])
        ind_kt2 = np.array([3,4,5])
        ind_kt3 = np.array([6,7,8])
        indkt = np.arange(0,num_nodes*k*9,9).reshape(-1,1)
        row_indices_kt1 = (np.tile(ind_kt1,num_nodes*k).reshape(num_nodes*k,len(ind_kt1)) +  indkt).flatten()
        row_indices_kt2 = (np.tile(ind_kt2,num_nodes*k).reshape(num_nodes*k,len(ind_kt2)) +  indkt).flatten()
        row_indices_kt3 = (np.tile(ind_kt3,num_nodes*k).reshape(num_nodes*k,len(ind_kt3)) +  indkt).flatten()
        self.declare_partials('K', 'kt1', rows=row_indices_kt1, cols=np.zeros(len(row_indices_kt1)).flatten())
        self.declare_partials('K', 'kt2', rows=row_indices_kt2, cols=np.zeros(len(row_indices_kt2)).flatten())
        self.declare_partials('K', 'kt3', rows=row_indices_kt3, cols=np.zeros(len(row_indices_kt3)).flatten())
        

        '''kappa'''
        ind_kp1 = np.array([0,1,2,3,6])
        ind_kp2 = np.array([1,3,4,5,7])
        ind_kp3 = np.array([2,5,6,7,8])
        indkp = np.arange(0,num_nodes*k*9,9).reshape(-1,1)
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
       
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kt1 = inputs['kt1']
        kt2 = inputs['kt2']
        kt3 = inputs['kt3']
        
        # compute tensor
        K = np.zeros((num_nodes,k,3,3))
       
        T_kb = np.ones((num_nodes,k,3))
        T_kb[:,:,0] = T_kb[:,:,0] * kb1
        T_kb[:,:,1] = T_kb[:,:,1] * kb2
        T_kb[:,:,2] = T_kb[:,:,2] * kb3
        K[:,:,0,:] = T_kb * kb1/kt1  
        K[:,:,1,:] = T_kb * kb2/kt2
        K[:,:,2,:] = T_kb * kb3/kt3
        
        outputs['K'] = K


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        # kappa = inputs['kappa']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        kt1 = inputs['kt1']
        kt2 = inputs['kt2']
        kt3 = inputs['kt3']

        Pk_pkb1 = np.zeros((num_nodes*k,9))
        Pk_pkb1[:, 0] = 2*kb1/kt1
        Pk_pkb1[:, 1] = kb2/kt1
        Pk_pkb1[:, 2] = kb3/kt1
        Pk_pkb1[:, 3] = kb2/kt2
        Pk_pkb1[:, 6] = kb3/kt3
        

        Pk_pkb2 = np.zeros((num_nodes*k,9))
        Pk_pkb2[:, 1] = kb1/kt1
        Pk_pkb2[:, 3] = kb1/kt2
        Pk_pkb2[:, 4] = 2*kb2/kt2
        Pk_pkb2[:, 5] = kb3/kt2
        Pk_pkb2[:, 7] = kb3/kt3

        Pk_pkb3 = np.zeros((num_nodes*k,9))
        Pk_pkb3[:, 2] = kb1/kt1
        Pk_pkb3[:, 5] = kb2/kt2
        Pk_pkb3[:, 6] = kb1/kt3
        Pk_pkb3[:, 7] = kb2/kt3
        Pk_pkb3[:, 8] = 2*kb3/kt3
        ''' kt '''
       
        Pk_pkt1 = np.zeros((num_nodes*k,3))
        Pk_pkt1[:, 0] = -kb1**2/kt1**2
        Pk_pkt1[:, 1] = -kb1*kb2/kt1**2
        Pk_pkt1[:, 2] = -kb1*kb3/kt1**2

        Pk_pkt2 = np.zeros((num_nodes*k,3))
        Pk_pkt2[:, 0] = -kb1*kb2/kt2**2
        Pk_pkt2[:, 1] = -kb2**2/kt2**2
        Pk_pkt2[:, 2] = -kb2*kb3/kt2**2
      
        Pk_pkt3 = np.zeros((num_nodes*k,3))
        Pk_pkt3[:, 0] = -kb1*kb3/kt3**2
        Pk_pkt3[:, 1] = -kb2*kb3/kt3**2
        Pk_pkt3[:, 2] = -kb3**2/kt3**2
 
        
        partials['K','kb1'][:] =  Pk_pkb1.flatten()
        partials['K','kb2'][:] =  Pk_pkb2.flatten()
        partials['K','kb3'][:] =  Pk_pkb3.flatten()
        partials['K','kt1'][:] =  Pk_pkt1.flatten()
        partials['K','kt2'][:] =  Pk_pkt2.flatten()
        partials['K','kt3'][:] =  Pk_pkt3.flatten()
        



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    
    comp.add_output('kb1', val=0.1)
    comp.add_output('kb2', val=10)
    comp.add_output('kb3', val=2)
    comp.add_output('kt1', val=1)
    comp.add_output('kt2', val=4)
    comp.add_output('kt3', val=1)
   
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TensorComp()
    group.add_subsystem('tensorcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)