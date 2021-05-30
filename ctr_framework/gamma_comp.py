import numpy as np
from openmdao.api import ExplicitComponent


class GammaComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=1, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        

        #Inputs

        self.add_input('psi', shape=(num_nodes,k,3))
        self.add_input('angle_eq',shape=(num_nodes,k))
        # outputs
        self.add_output('gamma',shape=(num_nodes,k,3))
        



        # partials

        
        row_indices = np.arange(num_nodes*k*3)
        col_indices = np.outer(np.arange(num_nodes*k),np.ones(3)).flatten()
        
        
        
        
        # print()
        self.declare_partials('gamma', 'angle_eq',rows=row_indices,cols=col_indices)
        self.declare_partials('gamma', 'psi')

        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        psi = inputs['psi']
        angle_eq = inputs['angle_eq']
        
        gamma = angle_eq[:,:,np.newaxis] - psi



        outputs['gamma'] = gamma
        
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']

        partials['gamma','psi'][:] = -np.identity(num_nodes*k*3)
        partials['gamma','angle_eq'][:] = 1
        
        

        



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 3
    k = 2
    comp = IndepVarComp()
    u = np.random.random((n,k,3))
    comp.add_output('psi', val=u)
    comp.add_output('angle_eq', val=np.random.rand(n,k))
    
    

    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = GammaComp(num_nodes=n,k=k)
    group.add_subsystem('Kappaequilcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    