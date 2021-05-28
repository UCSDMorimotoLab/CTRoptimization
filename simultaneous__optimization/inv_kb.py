import numpy as np
from openmdao.api import ExplicitComponent


class invkbComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes',default=3, types=int)

        

    def setup(self):
        #Inputs

        num_nodes = self.options['num_nodes']
        k = self.options['k']
        self.add_input('initial_condition_dpsi',shape=(k,3))
        self.add_output('objective')
        
        

        # partials
        # define indices
        # self.declare_partials('*', '*', method='fd')
        self.declare_partials('objective','initial_condition_dpsi')
        # self.declare_partials('objective','tube_ends_tip')

        
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        initial_condition_dpsi = inputs['initial_condition_dpsi']
        

        # temp[:,:,1] = (np.tanh(np.outer(np.arange(num_nodes),np.ones(k))-tube_ends_tip[:,1])/2 + 0.5)
        # temp[:,:,2] = (np.tanh(np.outer(np.arange(num_nodes),np.ones(k))-tube_ends_tip[:,2])/2 + 0.5)

        # idx0 = np.where(tube_ends_tip == 0)
        # idx1 = np.where(tube_ends_tip == 1)

        # tube_ends_tip[idx0] = 1
        # tube_ends_tip[idx1] = 0
        # penalize = np.zeros((num_nodes,k,3))
        # penalize = dpsi_ds * tube_ends_tip
                

        outputs['objective'] = np.linalg.norm(initial_condition_dpsi)
        
        

        
        

    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        initial_condition_dpsi = inputs['initial_condition_dpsi']
        ''' Jacobian of partial derivatives for P Pdot matrix.'''

        
        '''Computing Partials'''
        
        sumdpsi = sum(sum(initial_condition_dpsi**2))
        dob_dp = ((sumdpsi)**-0.5)* initial_condition_dpsi
        
        partials['objective','initial_condition_dpsi'] =  dob_dp.flatten()
        



        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    n = 3
    k = 3
    comp.add_output('initial_condition_dpsi',val=np.ones((k,3)))

    

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = invkbComp()
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    prob.model.list_outputs()
    
    
