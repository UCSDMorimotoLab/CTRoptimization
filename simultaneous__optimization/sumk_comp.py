import numpy as np
from openmdao.api import ExplicitComponent


class SumkComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('E', default=80.0, types=float)
        self.options.declare('J', default=80.0, types=float)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('tube_ends', shape=(num_nodes,k,3))
        self.add_input('kb1')
        self.add_input('kb2')
        self.add_input('kb3')
        self.add_input('K',shape=(num_nodes,k,3,3))


        # outputs
        self.add_output('K_s',shape=(num_nodes,k,3,3))


        # partials

        ''' kb '''
        # ind_kb1 = np.arange(0,9,1)
        # indkb = np.arange(0,num_nodes*k*9,9).reshape(-1,1)
        # row_indices_kb = (np.tile(ind_kb1,num_nodes*k).reshape(num_nodes*k,len(ind_kb1)) +  indkb).flatten()
        # self.declare_partials('K_s', 'kb1', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        # self.declare_partials('K_s', 'kb2', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        # self.declare_partials('K_s', 'kb3', rows=row_indices_kb, cols=np.zeros(len(row_indices_kb)).flatten())
        self.declare_partials('K_s', 'kb1')
        self.declare_partials('K_s', 'kb2')
        self.declare_partials('K_s', 'kb3')


        '''K'''
        row_indices_K=np.arange(num_nodes*k*3*3).flatten()
        col_indices_K = np.arange(num_nodes*k*3*3).flatten()
        self.declare_partials('K_s', 'K',rows= row_indices_K, cols=col_indices_K)
        

        '''tube_ends'''
        self.declare_partials('K_s', 'tube_ends')
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        # tube_section_length = inputs['tube_section_length']
        # beta = inputs['beta']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        K = inputs['K']
        tube_ends = inputs['tube_ends']

        # tube1 = kb1 * tube_ends[:,:,0]
        # tube2 = kb2 * tube_ends[:,:,1]
        # tube3 = kb3 * tube_ends[:,:,2]

        # sum_kb = tube1 + tube2 + tube3
        # K_s = np.zeros((num_nodes,k,3,3))
        
        tube_in_link = np.sum(tube_ends,2)
        
        idx1 = np.where((tube_in_link>0) & (tube_in_link<=1))
        idx2 = np.where((tube_in_link<=2.0) & (tube_in_link>1.0)) 
        idx3 = np.where((tube_in_link<=3.0) & (tube_in_link>2.0))
        

        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        idx3 = np.array(idx3)

        self.idx1 = idx1
        self.idx2 = idx2
        self.idx3 = idx3

        K_s = np.zeros((num_nodes,k,3,3))
        # K_s[idx1,:,:] = K[idx1,:,:] / (kb1)
        # K_s[idx2,:,:] = K[idx2,:,:] / (kb1+kb2)
        # K_s[idx3,:,:] = K[idx3,:,:] / (kb1+kb2+kb3)
        K_s[idx1[0,:],idx1[1,:],:,:] = K[idx1[0,:],idx1[1,:],:,:] / (kb1)
        K_s[idx2[0,:],idx2[1,:],:,:] = K[idx2[0,:],idx2[1,:],:,:] / (kb1+kb2)
        K_s[idx3[0,:],idx3[1,:],:,:] = K[idx3[0,:],idx3[1,:],:,:] / (kb1+kb2+kb3)
        
        outputs['K_s'] = K_s
        
        
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        tube_ends = inputs['tube_ends']
      
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        K = inputs['K']

        
        '''K'''
        #tube_in_link = np.sum(tube_ends,2)
        # print(tube_in_link.shape)
        # idx1 = np.where((tube_in_link>=0) & (tube_in_link<=1))
        # idx2 = np.where((tube_in_link<=2.0) & (tube_in_link>1.0)) 
        # idx3 = np.where((tube_in_link<=3.0) & (tube_in_link>2.0))
        # # print(idx1)
        # idx1 = np.array(idx1)
        # idx2 = np.array(idx2)
        # idx3 = np.array(idx3)
        idx1 = self.idx1
        idx2 = self.idx2
        idx3 = self.idx3


        

        
        Pks_pk = np.zeros((num_nodes,k,3,3))
        Pks_pk[idx1[0,:],idx1[1,:],:,:] = 1/(kb1)
        Pks_pk[idx2[0,:],idx2[1,:],:,:] = 1/(kb1+kb2)
        Pks_pk[idx3[0,:],idx3[1,:],:,:] = 1/(kb1+kb2+kb3)


        partials['K_s','K'][:] =  Pks_pk.flatten()

        '''kb'''
        # kb1
        
        Pk_pkb1 = np.zeros((num_nodes,k,3,3))
        Pk_pkb1[idx1[0,:],idx1[1,:],:,:] = (-1/kb1**2) * K[idx1[0,:],idx1[1,:],:,:]
        Pk_pkb1[idx2[0,:],idx2[1,:],:,:] = (-1/(kb1 + kb2)**2) * K[idx2[0,:],idx2[1,:],:,:]
        Pk_pkb1[idx3[0,:],idx3[1,:],:,:] = (-1/(kb1 + kb2 + kb3)**2) * K[idx3[0,:],idx3[1,:],:,:]
        
        

        # kb2
        Pk_pkb2 = np.zeros((num_nodes,k,3,3))
        Pk_pkb2[idx1[0,:],idx1[1,:],:,:] = 0
        Pk_pkb2[idx2[0,:],idx2[1,:],:,:] = (-1/(kb1 + kb2)**2) * K[idx2[0,:],idx2[1,:],:,:]
        Pk_pkb2[idx3[0,:],idx3[1,:],:,:] = (-1/(kb1 + kb2 + kb3)**2) * K[idx3[0,:],idx3[1,:],:,:]
        

        #kb3
        Pk_pkb3 = np.zeros((num_nodes,k,3,3))
        Pk_pkb3[idx1[0,:],idx1[1,:],:,:] = 0
        Pk_pkb3[idx2[0,:],idx2[1,:],:,:] = 0
        Pk_pkb3[idx3[0,:],idx3[1,:],:,:] = (-1/(kb1 + kb2 + kb3)**2) * K[idx3[0,:],idx3[1,:],:,:]
        
        partials['K_s','kb1'][:] =  Pk_pkb1.reshape((num_nodes*k*9,1))
        partials['K_s','kb2'][:] =  Pk_pkb2.reshape((num_nodes*k*9,1))
        partials['K_s','kb3'][:] =  Pk_pkb3.reshape((num_nodes*k*9,1))
        
        partials['K_s','tube_ends'] = 0


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 175
    k = 1
    comp = IndepVarComp()
    # comp.add_output('tube_section_length', val=0.1 * np.ones((2,3)))
    # comp.add_output('beta', val=0.1 * np.ones((2,3)))
    comp.add_output('kb1', val=1.65)
    comp.add_output('kb2', val=5.81)
    comp.add_output('kb3', val=70.35)
    comp.add_output('K', val = np.random.random((n,k,3,3)))
    tube_val = np.zeros((n,k,3))
    tube_val[0,:,0] = 1
    tube_val[1:,:,0] = 0

    tube_val[0,:,1] = 1
    tube_val[1,:,1] = 1
    tube_val[2,:,1] = 0

    tube_val[0,:,2] = 1
    tube_val[1,:,2] = 1
    tube_val[2,:,2] = 1

    comp.add_output('tube_ends', val = tube_val)
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = SumkComp(k=k,num_nodes=n)
    group.add_subsystem('sumkcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    