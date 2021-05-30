import numpy as np
from openmdao.api import ExplicitComponent


class DiameterComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)

        

    def setup(self):
        #Inputs

        self.add_input('d1')
        self.add_input('d2')
        self.add_input('d3')
        self.add_input('d4')
        self.add_input('d5')
        self.add_input('d6')

        self.add_output('diameterconstraint',shape=(1,3))

        # partials        
        self.declare_partials('diameterconstraint','d1')
        self.declare_partials('diameterconstraint','d2')
        self.declare_partials('diameterconstraint','d3')
        self.declare_partials('diameterconstraint','d4')
        self.declare_partials('diameterconstraint','d5')
        self.declare_partials('diameterconstraint','d6')
        
    def compute(self,inputs,outputs):
        
        d1 = inputs['d1']
        d2 = inputs['d2']
        d3 = inputs['d3']
        d4 = inputs['d4']
        d5 = inputs['d5']
        d6 = inputs['d6']
        constraint = np.zeros((1,3))
        constraint[:,0] = d2-d1
        constraint[:,1] = d4-d3
        constraint[:,2] = d6-d5
        

        outputs['diameterconstraint'] = constraint
        
        

    def compute_partials(self,inputs,partials):
        
        
        
        
        '''Computing Partials'''
        pdc_pd1 = np.zeros((3,1))
        pdc_pd1[0,:] = -1
        pdc_pd2 = np.zeros((3,1))
        pdc_pd2[0,:] = 1
        pdc_pd3 = np.zeros((3,1))
        pdc_pd3[1,:] = -1
        pdc_pd4 = np.zeros((3,1))
        pdc_pd4[1,:] = 1
        pdc_pd5 = np.zeros((3,1))
        pdc_pd5[2,:] = -1
        pdc_pd6 = np.zeros((3,1))
        pdc_pd6[2,:] = 1

        
        partials['diameterconstraint','d1'] = pdc_pd1
        partials['diameterconstraint','d2'] = pdc_pd2
        partials['diameterconstraint','d3'] = pdc_pd3 
        partials['diameterconstraint','d4'] = pdc_pd4
        partials['diameterconstraint','d5'] = pdc_pd5
        partials['diameterconstraint','d6'] = pdc_pd6


        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    comp.add_output('d1',val=1)
    comp.add_output('d2',val=2.5)
    comp.add_output('d3',val=3)
    comp.add_output('d4',val=4.5)
    comp.add_output('d5',val=5)
    comp.add_output('d6',val=6)

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = DiameterComp()
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    
    
