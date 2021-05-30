import numpy as np
from openmdao.api import ExplicitComponent


class PenalizeComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)
        self.options.declare('a', default=10, types=int)
        self.options.declare('num_nodes',default=3, types=int)

        

    def setup(self):
        #Inputs

        num_nodes = self.options['num_nodes']
        k = self.options['k']
        self.add_input('dpsi_ds',shape=(num_nodes,k,3))
        self.add_input('tube_ends_tip',shape=(k,3))
        self.add_output('penalized',shape=(num_nodes,k,3))
        

        # partials
        # define indices    
        # self.declare_partials('*', '*', method='fd')
        # row_indices_K = np.outer(np.arange(num_nodes*k),np.ones(3)).flatten()
        row_indices = np.outer(np.ones(3),np.arange(num_nodes*k)).flatten()
        col_indices = np.arange(num_nodes*k*3).flatten()
        # print(row_indices_K.shape)    s
        # print(col_indices_K.shape)

        self.declare_partials('penalized','dpsi_ds',rows=col_indices,cols=col_indices)
        row_indices_K = np.outer(np.ones(num_nodes),np.arange(3*k)).flatten()
        col_indices_K = np.arange(num_nodes*k*3).flatten()
        self.declare_partials('penalized','tube_ends_tip',rows=col_indices_K,cols=row_indices_K)
        # self.declare_partials('penalized','dpsi_ds')
        # self.declare_partials('penalized','tube_ends_tip')

        
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        a = self.options['a']
        dpsi_ds = inputs['dpsi_ds']
        tube_ends_tip = inputs['tube_ends_tip']
        # print(tube_ends_tip)
        temp = np.zeros((num_nodes,k,3)) 
        # a = 10
        temp[:,:,0] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends_tip[:,0]))/2 + 0.5) * dpsi_ds[:,:,0]
        temp[:,:,1] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends_tip[:,1]))/2 + 0.5) * dpsi_ds[:,:,1]
        temp[:,:,2] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends_tip[:,2]))/2 + 0.5) * dpsi_ds[:,:,2]
        
        # temp[:,:,1] = (np.tanh(np.outer(np.arange(num_nodes),np.ones(k))-tube_ends_tip[:,1])/2 + 0.5)
        # temp[:,:,2] = (np.tanh(np.outer(np.arange(num_nodes),np.ones(k))-tube_ends_tip[:,2])/2 + 0.5)

        # idx0 = np.where(tube_ends_tip == 0)
        # idx1 = np.where(tube_ends_tip == 1)

        # tube_ends_tip[idx0] = 1
        # tube_ends_tip[idx1] = 0
        # penalize = np.zeros((num_nodes,k,3))
        # penalize = dpsi_ds * tube_ends_tip
        # print(temp[:,0,2])
        
        outputs['penalized'] = temp
        
        

        
        

    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        a = self.options['a']
        tube_ends_tip = inputs['tube_ends_tip']
        dpsi_ds = inputs['dpsi_ds']
        
        '''Computing Partials'''
        # a = 10
        Pp_dpsi = np.zeros((num_nodes,k,3))
        # tanh(a*(x-xo))
        Pp_dpsi[:,:,0] = 0.5*np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k)) - tube_ends_tip[:,0])) + 0.5
        Pp_dpsi[:,:,1] = 0.5*np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k)) - tube_ends_tip[:,1])) + 0.5
        Pp_dpsi[:,:,2] = 0.5*np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k)) - tube_ends_tip[:,2])) + 0.5
                
        Pp_dt = np.zeros((num_nodes,k,3))
        # i = np.outer(np.arange(k),np.ones(3))
        # k_ = np.outer(np.ones(k),np.arange(3))
        # idx = 3*k*num_nodes-1+ k_*3 + i
        # print(-0.5*a*dpsi_ds[:,:,0]*(1 - np.tanh(a*(np.outer(np.arange(num_nodes),np.ones(k)) - tube_ends_tip[:,0]))**2))
        # # Pp_dt[idx.astype(int).flatten(),np.arange(k*3)] = 1. 
        Pp_dt[:,:,0] = -0.5*a*dpsi_ds[:,:,0]*(1 - np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k)) - tube_ends_tip[:,0]))**2)
        Pp_dt[:,:,1] = -0.5*a*dpsi_ds[:,:,1]*(1 - np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k)) - tube_ends_tip[:,1]))**2)
        Pp_dt[:,:,2] = -0.5*a*dpsi_ds[:,:,2]*(1 - np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k)) - tube_ends_tip[:,2]))**2)
        
        partials['penalized','dpsi_ds'][:] =  Pp_dpsi.flatten()
        partials['penalized','tube_ends_tip'][:] =  Pp_dt.flatten()
        

        


  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    n = 175
    k = 1
    
    comp.add_output('dpsi_ds',val=np.random.random((n,k,3)))
    comp.add_output('tube_ends_tip',val= np.random.random((k,3))*100)

    

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = PenalizeComp(num_nodes=n,k=k)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.model.list_outputs()