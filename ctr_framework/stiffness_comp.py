import numpy as np
from openmdao.api import ExplicitComponent

class StiffnessComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('E', default=80, types=int)
        self.options.declare('G12', default=30, types=int)
        self.options.declare('G3', default=80, types=int)

        self.options.declare('tube_nbr', default=3, types=int)
        

    '''This class is defining the CombinedModelPara P1-P5'''
    def setup(self):
        #Inputs
        self.add_input('d1')
        self.add_input('d2')
        self.add_input('d3')
        self.add_input('d4')
        self.add_input('d5')
        self.add_input('d6')

        # outputs
        self.add_output('kb1')
        self.add_output('kb2')
        self.add_output('kb3')
        self.add_output('kt1')
        self.add_output('kt2')
        self.add_output('kt3')



        # partials
        self.declare_partials('kb1','d1')
        self.declare_partials('kb1','d2')
        self.declare_partials('kb2','d3')
        self.declare_partials('kb2','d4')
        self.declare_partials('kb3','d5')
        self.declare_partials('kb3','d6')
        self.declare_partials('kt1','d1')
        self.declare_partials('kt1','d2')
        self.declare_partials('kt2','d3')
        self.declare_partials('kt2','d4')
        self.declare_partials('kt3','d5')
        self.declare_partials('kt3','d6')

        
    def compute(self,inputs,outputs):

        E = self.options['E']
        G12 = self.options['G12']
        G3 = self.options['G3']

        d1 = inputs['d1']
        d2 = inputs['d2']
        d3 = inputs['d3']
        d4 = inputs['d4']
        d5 = inputs['d5']
        d6 = inputs['d6']

        

        outputs['kb1'] = E*np.pi*(d2**4-d1**4)/64
        outputs['kb2'] = E*np.pi*(d4**4-d3**4)/64
        outputs['kb3'] = E*np.pi*(d6**4-d5**4)/64
        outputs['kt1'] = G12*np.pi*(d2**4-d1**4)/32
        outputs['kt2'] = G12*np.pi*(d4**4-d3**4)/32
        outputs['kt3'] = G3*np.pi*(d6**4-d5**4)/32



    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        E = self.options['E']
        G12 = self.options['G12']
        G3 = self.options['G3']

        d1 = inputs['d1']
        d2 = inputs['d2']
        d3 = inputs['d3']
        d4 = inputs['d4']
        d5 = inputs['d5']
        d6 = inputs['d6']
        # bending
        partials['kb1','d1'] = E*np.pi*(-4*d1**3)/64
        partials['kb1','d2'] = E*np.pi*(4*d2**3)/64
        partials['kb2','d3'] = E*np.pi*(-4*d3**3)/64
        partials['kb2','d4'] = E*np.pi*(4*d4**3)/64
        partials['kb3','d5'] = E*np.pi*(-4*d5**3)/64
        partials['kb3','d6'] = E*np.pi*(4*d6**3)/64
        # torsion
        partials['kt1','d1'] = G12*np.pi*(-4*d1**3)/32
        partials['kt1','d2'] = G12*np.pi*(4*d2**3)/32
        partials['kt2','d3'] = G12*np.pi*(-4*d3**3)/32
        partials['kt2','d4'] = G12*np.pi*(4*d4**3)/32
        partials['kt3','d5'] = G3*np.pi*(-4*d5**3)/32
        partials['kt3','d6'] = G3*np.pi*(4*d6**3)/32 


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    comp.add_output('d1', val=0.1)
    comp.add_output('d2', val=0.156)
    comp.add_output('d3', val=0.156)
    comp.add_output('d4', val=0.156)
    comp.add_output('d5', val=0.156)
    comp.add_output('d6', val=0.156)


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = StiffnessComp()
    group.add_subsystem('Stiffnesscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)