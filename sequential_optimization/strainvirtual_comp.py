import numpy as np
from openmdao.api import ExplicitComponent
import math


class StrainvirtualComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('num_t', default=2, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        num_t = self.options['num_t']

        #Inputs
        self.add_input('tube_ends_hyperbolic',shape=(num_nodes,k,3))
        self.add_input('strain',shape=(num_nodes,k,num_t,3))


        # outputs
        self.add_output('strain_virtual',shape=(num_nodes,k,num_t,3))
        


        # partials
        
        col_indices_b = np.outer(np.ones(num_nodes*k),np.outer(np.ones(num_t),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        col_indices_t = np.arange(num_nodes*k*num_t*3).flatten()
        self.declare_partials('strain_virtual','tube_ends_hyperbolic', rows = col_indices_t,cols=col_indices_b.flatten())

        self.declare_partials('strain_virtual', 'strain', rows= col_indices_t,cols= col_indices_t)


    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_ends_hyperbolic = inputs['tube_ends_hyperbolic']
        strain = inputs['strain']
        
        
        
        outputs['strain_virtual'] = strain * tube_ends_hyperbolic[:,:,np.newaxis,:]        


    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        num_t = self.options['num_t']
        k = self.options['k']
        tube_ends_hyperbolic = inputs['tube_ends_hyperbolic']
        strain = inputs['strain']
        """partials Jacobian of partial derivatives."""
            
        Psv_s = np.zeros((num_nodes,k,num_t,3))
        Psv_s[:,:,:,:] = tube_ends_hyperbolic[:,:,np.newaxis,:]

        

        partials['strain_virtual','strain'][:] = Psv_s.flatten()
        
        Psv_th = np.zeros((num_nodes,k,num_t,3))
        Psv_th = strain
        
        partials['strain_virtual','tube_ends_hyperbolic'][:] = strain.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n = 10
    k = 4
    idx = int(n/2) 
    hyper = np.random.random((n,k,3))
    comp.add_output('tube_ends_hyperbolic',val = hyper )
    beta_init = np.zeros((k,3))
    beta_init[:,0] = 3*n/4
    beta_init[:,1] = n/2
    beta_init[:,2] = n/8

    comp.add_output('strain', val=np.random.rand(n,k,2,3))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = StrainvirtualComp(num_nodes=n,k=k)
    group.add_subsystem('interpolationknComp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
  


