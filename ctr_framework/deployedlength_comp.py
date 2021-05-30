import numpy as np
from openmdao.api import ExplicitComponent


class DeployedlengthComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)

        

    def setup(self):
        #Inputs
        k = self.options['k']
        self.add_input('tube_section_length',shape=(1,3))
        self.add_input('beta',shape=(k,3))

        self.add_output('deployedlength12constraint',shape=(1,k))
        self.add_output('deployedlength23constraint',shape=(1,k))
        
        # partials
        # define indices
        row_indices = np.outer(np.arange(k),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(1),np.array([0,1,2])).flatten()) + (np.arange(0,k*3,3).reshape(-1,1))
        row_indices_t = np.outer(np.arange(k),np.ones(3)).flatten()
        col_indices_t = np.outer(np.ones(k),np.arange(3)).flatten()
        self.declare_partials('deployedlength12constraint','tube_section_length',rows=row_indices_t.flatten(),cols=col_indices_t.flatten())
        self.declare_partials('deployedlength12constraint','beta',rows=row_indices.flatten(),cols=col_indices.flatten())
        self.declare_partials('deployedlength23constraint','tube_section_length',rows=row_indices_t.flatten(),cols=col_indices_t.flatten())
        self.declare_partials('deployedlength23constraint','beta',rows=row_indices.flatten(),cols=col_indices.flatten())
        
    def compute(self,inputs,outputs):
        
        
        k = self.options['k']
        tube_section_length = inputs['tube_section_length']
        beta = inputs['beta']
        deployed_length = np.zeros((k,3))
        deployed_length = tube_section_length + beta
        constraint12 = np.zeros((1,k))
        constraint23 = np.zeros((1,k))
        constraint12 = deployed_length[:,0] - deployed_length[:,1]
        constraint23 = deployed_length[:,1] - deployed_length[:,2]

        outputs['deployedlength12constraint'] = np.reshape(constraint12,(1,k))
        outputs['deployedlength23constraint'] = np.reshape(constraint23,(1,k))
        

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

        
        partials['deployedlength12constraint','tube_section_length'] = pc12_pt.flatten()
        partials['deployedlength12constraint','beta'][:] = pc12_pb.flatten()
        partials['deployedlength23constraint','tube_section_length'] = pc23_pt.flatten()
        partials['deployedlength23constraint','beta'][:] = pc23_pb.flatten()

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    k=15
    comp.add_output('tube_section_length',val = [175,120,65])
    beta_init = np.zeros((k,3))
    beta_init[:,0] = -55
    beta_init[:,1] = -40
    beta_init[:,2] = -25

    comp.add_output('beta', val=beta_init)
    
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = DeployedlengthComp(k=k)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.model.list_outputs()
    
    
