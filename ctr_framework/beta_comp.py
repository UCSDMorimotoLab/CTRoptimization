import numpy as np
from openmdao.api import ExplicitComponent


class BetaComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)

        

    def setup(self):
        #Inputs
        k = self.options['k']
        self.add_input('beta',shape=(k,3))

        self.add_output('beta12constraint',shape=(1,k))
        self.add_output('beta23constraint',shape=(1,k))
        

        # partials
        # define indices
        row_indices = np.outer(np.arange(k),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(1),np.array([0,1,2])).flatten()) + (np.arange(0,k*3,3).reshape(-1,1))
        self.declare_partials('beta12constraint','beta',rows=row_indices.flatten(),cols=col_indices.flatten())
        self.declare_partials('beta23constraint','beta',rows=row_indices.flatten(),cols=col_indices.flatten())
        
    def compute(self,inputs,outputs):
        
        
        k = self.options['k']
        beta = inputs['beta']
        
        constraint12 = np.zeros((1,k))
        constraint23 = np.zeros((1,k))
        constraint12 = beta[:,0] - beta[:,1]
        constraint23 = beta[:,1] - beta[:,2]

        outputs['beta12constraint'] = np.reshape(constraint12,(1,k))
        outputs['beta23constraint'] = np.reshape(constraint23,(1,k))
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        k = self.options['k']
        
        '''Computing Partials'''
        pc12_pb = np.zeros((k,3))
        pc12_pb[:,0] = 1
        pc12_pb[:,1] = -1

        pc23_pb = np.zeros((k,3))
        pc23_pb[:,1] = 1
        pc23_pb[:,2] = -1

        pc12_pt = np.zeros((k,3))
        pc12_pt[:,0] = 1
        pc12_pt[:,1] = -1

        pc23_pt = np.zeros((k,3))
        pc23_pt[:,1] = 1
        pc23_pt[:,2] = -1

        
        
        partials['beta12constraint','beta'][:] = pc12_pb.flatten()
        partials['beta23constraint','beta'][:] = pc23_pb.flatten()



        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    k=1
    
    beta_init = np.zeros((k,3))
    beta_init[:,0] = -55
    beta_init[:,1] = -40
    beta_init[:,2] = -25

    comp.add_output('beta', val=beta_init)
    

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = BetaComp(k=k)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    prob.model.list_outputs()
    
    
