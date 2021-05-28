import numpy as np
from openmdao.api import ExplicitComponent


class OptpathComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_pt', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_pt = self.options['num_pt']
        k = self.options['k']

        #Inputs
        self.add_input('desptsconstraints', shape=(k,3))
        self.add_input('pt', shape=(num_pt,3))
        

        # outputs
        self.add_output('optpath_constraints',shape=(k,3))
        
        
        
        row_indices = np.arange(k*3)
        col_indices = np.outer(np.ones(k),np.array([0,1,2])).flatten() + np.outer(np.arange(0,num_pt*3,k*3),np.ones(3)).flatten()
        self.declare_partials('optpath_constraints', 'desptsconstraints')
        self.declare_partials('optpath_constraints', 'pt',rows=row_indices,cols=col_indices)

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_pt= self.options['num_pt']
        desptsconstraints = inputs['desptsconstraints']
        pt = inputs['pt']


        opt_cons = np.zeros((k,3))
        # idx = np.linspace(0, num_pt, k, endpoint=True)
        opt_cons = desptsconstraints - pt[::k,:]
        
        
        outputs['optpath_constraints'] = opt_cons


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        
        # Po_pt = np.zeros((k*3,num_pt*3))
        # Po_pt[np.arange(k*3),]
        

        partials['optpath_constraints','desptsconstraints'][:] = np.identity(k*3)
        partials['optpath_constraints','pt'][:]= -1

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=175
    k=10
    num_pt = 100 
    comp = IndepVarComp()
    comp.add_output('pt', val=np.random.random((num_pt,3)))
    comp.add_output('desptsconstraints', val=np.random.random((k,3)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = OptpathComp(num_pt=num_pt,k=k)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    # prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
