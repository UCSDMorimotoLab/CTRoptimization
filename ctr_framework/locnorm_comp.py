import numpy as np
from openmdao.api import ExplicitComponent


class LocnormComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        
        self.add_input('loc',shape=(k,3,1))
        # self.add_input('loc',shape=(2,1))
        # outputs
        self.add_output('locnorm')
        
        
        self.declare_partials('locnorm', 'loc')
       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        
        loc = inputs['loc']
        norm = np.linalg.norm(loc)
        outputs['locnorm'] = norm

        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        loc = inputs['loc']
        pln_pn = np.zeros((k,3,1))
        pln_pn[:,:] =  loc*(np.sum(loc**2)**-0.5)

        '''Computing Partials'''
        partials['locnorm','loc'][:]= pln_pn.flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=2
    k=3
    comp = IndepVarComp()
    
    comp.add_output('loc', val = np.random.random((k,3,1)))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = LocnormComp(num_nodes=n,k=k)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
