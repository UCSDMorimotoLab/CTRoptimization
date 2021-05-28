import numpy as np
from openmdao.api import ExplicitComponent

class FinalpointComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_cp', default=3, types=int)
        

    
    def setup(self):
        num_cp = self.options['num_cp']
        

        #Inputs
        self.add_input('cp', shape=(num_cp,3))

        # outputs
        self.add_output('finalpoint_constraint',shape=(1,3))
        self.declare_partials('finalpoint_constraint', 'cp')
        

       
        
    def compute(self,inputs,outputs):

        
        num_cp = self.options['num_cp']
        cp = inputs['cp']
        
        outputs['finalpoint_constraint'] = np.reshape(cp[-1,:],(1,3))


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_cp = self.options['num_cp']
        cp = inputs['cp']
        
        '''Computing Partials'''
        pf_pcp = np.zeros((3,num_cp*3))
        pf_pcp[:,(num_cp-1)*3:] = np.identity(3)
    
        partials['finalpoint_constraint','cp'][:]= pf_pcp

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    num_cp=39
   
    comp = IndepVarComp()
    comp.add_output('cp', val=np.random.random((num_cp,3)))
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = FinalpointComp(num_cp=num_cp)
    group.add_subsystem('startpointcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    # prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
