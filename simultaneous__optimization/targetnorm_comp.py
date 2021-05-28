import numpy as np
from openmdao.api import ExplicitComponent



class TargetnormComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        

        

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        


        #Inputs
        self.add_input('targetpoints',shape=(k,3))
        

        # outputs
        self.add_output('targetnorm',shape=(k,1))


        # partials
        row_indices = np.outer(np.arange(k),np.ones(3))
        col_indices = np.arange(k*3)
        self.declare_partials('targetnorm', 'targetpoints',rows=row_indices.flatten(),cols=col_indices.flatten())
        
        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        targetpoints = inputs['targetpoints']
        magnitude = np.linalg.norm(targetpoints,axis=1)
        
        outputs['targetnorm'] = magnitude.reshape(-1,1)



    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        targetpoints = inputs['targetpoints']
        sumdpsi = np.sum(targetpoints**2,1)
        pt_pt = np.zeros((k,3))
        pt_pt[:,0] = (((sumdpsi)**-0.5) * targetpoints[:,0])
        pt_pt[:,1] = (((sumdpsi)**-0.5) * targetpoints[:,1])
        pt_pt[:,2] = (((sumdpsi)**-0.5) * targetpoints[:,2])
        partials['targetnorm','targetpoints'][:] = pt_pt.flatten()
       
    


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 175
    k = 2
    comp = IndepVarComp()
    comp.add_output('targetpoints', val = np.random.random((k,3))*100)

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TargetnormComp(k=k,num_nodes=n)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    