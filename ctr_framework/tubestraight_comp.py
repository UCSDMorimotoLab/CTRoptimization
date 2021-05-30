import numpy as np
from openmdao.api import ExplicitComponent


class TubestraightComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)
        

    def setup(self):
        #Inputs


        self.add_input('tube_section_length',shape=(1,3))
        self.add_input('tube_section_straight',shape=(1,3))
        
        self.add_output('tubestraightconstraint',shape=(1,3))
        

        # partials
        self.declare_partials('tubestraightconstraint','tube_section_length')
        self.declare_partials('tubestraightconstraint','tube_section_straight')
        
        
    def compute(self,inputs,outputs):
        
        
        tube_section_straight = inputs['tube_section_straight']
        tube_section_length = inputs['tube_section_length']
        
        constraint = np.zeros((1,3))
        constraint = tube_section_length - tube_section_straight 


        outputs['tubestraightconstraint'] = constraint
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        tube_section_straight = inputs['tube_section_straight']
        tube_section_length = inputs['tube_section_length']
        
        '''Computing Partials'''
        
        
        partials['tubestraightconstraint','tube_section_length'] = np.identity(3)
        partials['tubestraightconstraint','tube_section_straight'] = -np.identity(3)
        



        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    
    comp.add_output('tube_section_length',val=np.random.random((1,3))*10)
    comp.add_output('tube_section_straight',val=np.random.random((1,3)))
    

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = TubestraightComp()
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    
    
