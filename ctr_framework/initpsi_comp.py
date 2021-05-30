import numpy as np
from openmdao.api import ExplicitComponent


class InitialpsiComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=40, types=int)
        self.options.declare('k', default=3, types=int)

        

    def setup(self):
        #Inputs
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        self.add_input('initial_condition_dpsi',shape=(k,3))
        self.add_input('alpha',shape=(k,3))
        self.add_input('beta',shape=(k,3))
 
        self.add_output('initial_condition_psi',shape=((k,3)))

        # partials
        # define indices
        self.declare_partials('initial_condition_psi','alpha',rows=np.arange(k*3) , cols=np.arange(k*3))
        self.declare_partials('initial_condition_psi','beta',rows=np.arange(k*3) , cols=np.arange(k*3))
        self.declare_partials('initial_condition_psi','initial_condition_dpsi')
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        alpha = inputs['alpha']
        beta = inputs['beta']
        initial_condition_dpsi = inputs['initial_condition_dpsi']
        
        init = np.zeros((k,3))
        init = alpha - beta * initial_condition_dpsi
        outputs['initial_condition_psi'] = init
        
        

    def compute_partials(self,inputs,partials):
        
        
        ''' Jacobian of partial derivatives for P Pdot matrix.'''
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        initial_condition_dpsi = inputs['initial_condition_dpsi']
        beta = inputs['beta']
        alpha = inputs['alpha']
        
        
        '''Computing Partials'''
        Pi_pa = np.zeros((k,3))
        Pi_pa[:,0] = 1
        Pi_pa[:,1] = 1
        Pi_pa[:,2] = 1
        partials['initial_condition_psi','alpha']= Pi_pa.flatten()

        Pi_pb = np.zeros((k,3))
        Pi_pb[:,0] = -initial_condition_dpsi[:,0]
        Pi_pb[:,1] = -initial_condition_dpsi[:,1]
        Pi_pb[:,2] = -initial_condition_dpsi[:,2]
        partials['initial_condition_psi','beta']= Pi_pb.flatten()

        Pi_pdpsi = np.zeros((3*k, k*3))
        Pi_pdpsi[:3*k,:3*k] = -np.diag(beta.flatten())
        partials['initial_condition_psi','initial_condition_dpsi']= Pi_pdpsi

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()
    num_nodes = 35
    k = 1
  
    dpsi = np.random.random((k,3))
    al =  np.random.random((k,3))
    be =  np.random.random((k,3))

    comp.add_output('alpha',val = al)
    comp.add_output('beta',val = be)
    comp.add_output('initial_condition_dpsi',val=dpsi)
  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = InitialpsiComp(num_nodes=num_nodes,k=k)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    # prob.check_partials(compact_print=True)
    prob.model.list_outputs()
