import numpy as np
from openmdao.api import ExplicitComponent

class PathobjectiveComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_cp', default=3, types=int)
        self.options.declare('r2')
        self.options.declare('r1')
        
    
    def setup(self):
        num_cp = self.options['num_cp']
        r2 = self.options['r2']
        r1 = self.options['r1']
        

        #Inputs
        self.add_input('path_obj1')
        self.add_input('path_obj2')
        

        # outputs
        self.add_output('path_objective')
        self.declare_partials('path_objective', 'path_obj1')
        self.declare_partials('path_objective', 'path_obj2')
        

       
        
    def compute(self,inputs,outputs):

        r2 = self.options['r2']
        r1 = self.options['r1']
        num_cp = self.options['num_cp']
        path_obj1 = inputs['path_obj1']
        path_obj2 = inputs['path_obj2']
        
        
        
        self.r1 = r1
        self.r2 = r2
        path_objective = r1 * path_obj1 + r2 * path_obj2 
        
        outputs['path_objective'] = path_objective
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_cp = self.options['num_cp']
        r2 = self.options['r2']
        r1 = self.options['r1']
        
        '''Computing Partials'''
        r1 = self.r1
        r2 = self.r2
        
        

        
        partials['path_objective','path_obj1'][:]= r1
        partials['path_objective','path_obj2'][:]= r2
        

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    num_cp=50
    r2 = 1
    r1 = 2
    comp = IndepVarComp()
    comp.add_output('path_obj1', val=10)
    comp.add_output('path_obj2', val=100)
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = PathobjectiveComp(num_cp=num_cp,r2=r2,r1=r1)
    group.add_subsystem('startpointcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
