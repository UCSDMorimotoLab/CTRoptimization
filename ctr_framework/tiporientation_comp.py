import numpy as np
from openmdao.api import ExplicitComponent

class TiporientationComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)
        self.options.declare('tar_vector')
        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('desptsconstraints',shape=(k,3))

        # outputs
        self.add_output('tiporientation',shape=(k))
        
        row_indices = np.outer(np.arange(k),np.ones(3)).flatten()
        col_indices = np.arange(k*3)
        
        self.declare_partials('tiporientation', 'desptsconstraints',rows=row_indices,cols=col_indices)
        

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        tar_vector = self.options['tar_vector']
        desptsconstraints = inputs['desptsconstraints']
        
        dot = (desptsconstraints - tar_vector[:,0]) @  (tar_vector[:,1] - tar_vector[:,0])
        
        outputs['tiporientation'] = dot


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        tar_vector = self.options['tar_vector']
        
        '''Computing Partials'''
        pd_pp = np.zeros((k,3))
        pd_pp[:,:] = (tar_vector[:,1] - tar_vector[:,0]).T

        partials['tiporientation','desptsconstraints'][:] = pd_pp.flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=1
    k=10
    tar_vector = np.random.rand(3,2)
    comp = IndepVarComp()
    comp.add_output('desptsconstraints', val=np.random.random((k,3)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TiporientationComp(num_nodes=n,k=k,tar_vector=tar_vector)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
