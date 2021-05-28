import numpy as np
from openmdao.api import ExplicitComponent


class BcComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=3, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        self.add_input('dpsi_ds',shape=(num_nodes,k,3))


        self.add_output('torsionconstraint',shape=((k,3)))

        # partials
        self.declare_partials('torsionconstraint','dpsi_ds')
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        dpsi_ds = inputs['dpsi_ds']
        bc = np.zeros((k,3))
    
        bc[:,:] = dpsi_ds[-1,:,:]
        
        outputs['torsionconstraint'] = bc
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        dpsi_ds = inputs['dpsi_ds']
        
        '''Computing Partials'''
        ppsi_dot = np.zeros((3*k, num_nodes*k*3))
        ppsi_dot[:,(num_nodes-1)*k*3:] = np.identity(k*3)
    
        partials['torsionconstraint','dpsi_ds']= ppsi_dot


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes =3
    k = 1
  
    tip=np.random.random((k,3))*3
    dpsi=np.random.random((num_nodes,k,3))
    comp.add_output('tube_ends_tip',val=tip)
    comp.add_output('dpsi_ds',val=dpsi)
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = BcComp(k=k,num_nodes=num_nodes)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.model.list_outputs()
    
    
