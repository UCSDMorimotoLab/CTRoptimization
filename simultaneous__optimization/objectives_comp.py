import numpy as np
from openmdao.api import ExplicitComponent

class ObjectivesComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_cp', default=3, types=int)
        # self.options.declare('r2')
        # self.options.declare('r3')
        

        

    
    def setup(self):
        num_cp = self.options['num_cp']
        # r2 = self.options['r2']
        # r3 = self.options['r3']
        

        #Inputs
        # self.add_input('obj1')
        self.add_input('objective_tj')
        self.add_input('objective4')

        # outputs
        self.add_output('objectives')
        # self.declare_partials('objectives', 'obj1')
        self.declare_partials('objectives', 'objective_tj')
        self.declare_partials('objectives', 'objective4')

       
        
    def compute(self,inputs,outputs):

        
        
        # obj1 = inputs['obj1']
        objective_tj = inputs['objective_tj']
        objective4 = inputs['objective4']
    
        
        # outputs['objectives'] = obj1 + objective_tj 
        outputs['objectives'] = objective4 + objective_tj * 0
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_cp = self.options['num_cp']
       
        
        
        # partials['objectives','obj1'][:]= 1
        partials['objectives','objective_tj'][:]= 1 * 0
        partials['objectives','objective4'][:]= 1

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    num_cp=50
    r2 = 1
    r1 = 2
    comp = IndepVarComp()
    # comp.add_output('obj1', val=10)
    comp.add_output('obective4', val=10)
    comp.add_output('objective_tj', val=100)
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ObjectivesComp(num_cp=num_cp)
    group.add_subsystem('startpointcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
