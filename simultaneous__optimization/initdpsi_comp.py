import numpy as np
from openmdao.api import ExplicitComponent


class InitialdpsiComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=3, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        self.add_input('dpsi_ds',shape=(num_nodes,k,3))

        
        # outputs
        # row_idx = np.outer(np.arange(k*3),np.ones(k*3*num_nodes))
        # row_idx = np.arange(num_nodes*k*3)
        # col_idx = np.outer(np.ones(k*3),np.arange(3*k*num_nodes))

        self.add_output('initial_condition_dpsi',shape=((k,3)))

        # partials
        # define indices
        
        self.declare_partials('initial_condition_dpsi','dpsi_ds')
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        dpsi_ds = inputs['dpsi_ds']
        outputs['initial_condition_dpsi'] = dpsi_ds[0,:,:]
        # print(outputs['initial_condition_dpsi'])
        
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        dpsi_ds = inputs['dpsi_ds']
       
        
        '''Computing Partials'''

        Pi_pdpsi = np.zeros((3*k, num_nodes*k*3))
        Pi_pdpsi[:3*k,:3*k] = np.diag(np.ones((3*k)))
        partials['initial_condition_dpsi','dpsi_ds']= Pi_pdpsi
       


        # print("idx=",idx)
        # print()
        # print(idx.shape)
        
        # for i in range(k*3):
        #     print(idx.flatten()[i])
        #     ppsi_dot[i,int((idx.flatten())[i])] = 1.
        
        # ppsi_dot[idx.astype(int),np.arange(num_nodes*k*3)] = 1.
        
        # partials['torsionconstraint','dpsi_ds']= ppsi_dot
        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes = 10
    k = 3
  
    dpsi = np.random.random((num_nodes,k,3))
    

    
    comp.add_output('dpsi_ds',val=dpsi)
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = InitialdpsiComp()
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    prob.model.list_outputs()
