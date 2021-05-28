import numpy as np
from openmdao.api import ExplicitComponent


class InitialRComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=2, types=int)
        self.options.declare('k', default=3, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        self.add_input('psi',shape=(num_nodes,k,3))
        

        
        # outputs
        self.add_output('initial_condition_R',shape=((k,3,3)))

        # partials
        # define indices
        self.declare_partials('initial_condition_R','psi')
        
    def compute(self,inputs,outputs):
        
        
        k = self.options['k']
        psi = inputs['psi']
        R = np.zeros((k,3,3))
        R[:,0,0] = np.cos(psi[0,:,0])
        R[:,0,1] = -np.sin(psi[0,:,0])
        R[:,1,0] = np.sin(psi[0,:,0])
        R[:,1,1] = np.cos(psi[0,:,0])
        R[:,2,2] = np.ones(k)


        outputs['initial_condition_R'] = Ｒ
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        psi = inputs['psi']

        
        
        '''Computing Partials'''
        
        Pr_ppsi = np.zeros((k*3*3, num_nodes*k*3))
        k_idx = np.arange(k)
        n_idx = np.arange(k)
        Pr_ppsi[k_idx*9,n_idx*3] = -np.sin(psi[0,:,0])
        Pr_ppsi[k_idx*9+1,n_idx*3] = -np.cos(psi[0,:,0])
        Pr_ppsi[k_idx*9+3,n_idx*3] = np.cos(psi[0,:,0])
        Pr_ppsi[k_idx*9+4,n_idx*3] = -np.sin(psi[0,:,0])
                
        partials['initial_condition_R','psi'][:]= Pr_ppsi
        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes = 20
    k = 18
  
    
    

    comp.add_output('psi',val = np.random.random((num_nodes,k,3)))
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = InitialRComp(num_nodes=num_nodes,k=k)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
