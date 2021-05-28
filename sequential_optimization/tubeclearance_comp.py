import numpy as np
from openmdao.api import ExplicitComponent


class TubeclearanceComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)

        

    def setup(self):
        #Inputs


        self.add_input('d2')
        self.add_input('d3')
        self.add_input('d4')
        self.add_input('d5')

        self.add_output('tubeclearanceconstraint',shape=(1,2))
        

        # partials 
        self.declare_partials('tubeclearanceconstraint','d2')
        self.declare_partials('tubeclearanceconstraint','d3')
        self.declare_partials('tubeclearanceconstraint','d4')
        self.declare_partials('tubeclearanceconstraint','d5')
        
    def compute(self,inputs,outputs):
        
        
        d2 = inputs['d2']
        d3 = inputs['d3']
        d4 = inputs['d4']
        d5 = inputs['d5']
        constraint = np.zeros((1,2))
        constraint[:,0] = d3-d2
        constraint[:,1] = d5-d4
        

        outputs['tubeclearanceconstraint'] = constraint
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        
        '''Computing Partials'''
        pdc_pd2 = np.zeros((2,1))
        pdc_pd2[0,:] = -1
        pdc_pd3 = np.zeros((2,1))
        pdc_pd3[0,:] = 1
        pdc_pd4 = np.zeros((2,1))
        pdc_pd4[1,:] = -1
        pdc_pd5 = np.zeros((2,1))
        pdc_pd5[1,:] = 1
        
        partials['tubeclearanceconstraint','d2'] = pdc_pd2
        partials['tubeclearanceconstraint','d3'] = pdc_pd3 
        partials['tubeclearanceconstraint','d4'] = pdc_pd4
        partials['tubeclearanceconstraint','d5'] = pdc_pd5



        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    
    comp.add_output('d2',val=2.5)
    comp.add_output('d3',val=3)
    comp.add_output('d4',val=5)
    comp.add_output('d5',val=6)

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = TubeclearanceComp()
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    
    
    
