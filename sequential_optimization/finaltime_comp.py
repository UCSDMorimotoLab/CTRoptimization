import numpy as np
from openmdao.api import ExplicitComponent


class FinaltimeComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']

        self.add_input('tube_section_length',shape=(1,3))


        self.add_output('final_time')

        # partials
        
        self.declare_partials('final_time','tube_section_length')
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        tube_section_length = inputs['tube_section_length']
        outputs['final_time'] = tube_section_length[:,0]
        
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        num_nodes = self.options['num_nodes']
       
        
        '''Computing Partials'''

        pf_pt = np.zeros((1,3))
        pf_pt[:,0] = 1
        partials['final_time','tube_section_length'][:] = pf_pt  



if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes = 10
    k = 3
  
    tube = np.random.random((1,3))
    

    
    comp.add_output('tube_section_length',val=tube)
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = FinaltimeComp()
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.model.list_outputs()
