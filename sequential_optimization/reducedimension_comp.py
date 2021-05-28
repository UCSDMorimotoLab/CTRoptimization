import numpy as np
from openmdao.api import ExplicitComponent


class ReducedimensionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)
        self.options.declare('num_t', default=2, types=int)
        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        num_t = self.options['num_t']
        k = self.options['k']
        

        #Inputs
        self.add_input('strain_virtual', shape=(num_nodes,k,num_t,3))
        
        # outputs
        self.add_output('strain_dim1',shape=(num_nodes*k*3*num_t,1))
        
        
       
        self.declare_partials('strain_dim1', 'strain_virtual')
        
       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        num_t= self.options['num_t']
        strain_virtual = inputs['strain_virtual']
        

        outputs['strain_dim1'] = strain_virtual.reshape(-1,1)
        # outputs['strain_dim1'] = strain_virtual.flatten()
                


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        k = self.options['k']
        num_nodes= self.options['num_nodes']
        num_t= self.options['num_t']

        partials['strain_dim1','strain_virtual'][:]= np.identity(num_nodes*k*num_t*3)

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=9
    k=7
    comp = IndepVarComp()
    comp.add_output('strain_virtual', val=np.random.random((n,k,2,3)))
    
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    base = np.random.random((3,1))
    comp = ReducedimensionComp(num_nodes=n,k=k)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
