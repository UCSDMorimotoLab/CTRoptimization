import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class ReachtargetptsComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('k', default=3, types=int)
        self.options.declare('targets')
        

        

    
    def setup(self):
        k = self.options['k']

        #Inputs
        self.add_input('desptsconstraints', shape=(k,3))

        # outputs
        self.add_output('targetpoints',shape=(k,3))
        
       
        self.declare_partials('targetpoints','desptsconstraints')

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        targets = self.options['targets']
        desptsconstraints = inputs['desptsconstraints']
        error = desptsconstraints -  targets 
        self.error = error
        
        outputs['targetpoints'] = error


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        k = self.options['k']
       
        '''Computing Partials'''
        pt_pp = np.zeros((k,3))
        partials['targetpoints','desptsconstraints'][:] = np.identity(k*3)

        
if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=175
    k=5
    targets = np.random.random((k,3))
    comp = IndepVarComp()
    comp.add_output('desptsconstraints', val=np.random.random((k,3))*100)
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ReachtargetptsComp(k=k,targets=targets)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
