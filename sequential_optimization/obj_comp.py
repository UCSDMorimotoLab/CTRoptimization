import numpy as np
from openmdao.api import ExplicitComponent


class ObjComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes',default=3, types=int)

        

    def setup(self):
        #Inputs

        num_nodes = self.options['num_nodes']
        k = self.options['k']
        self.add_input('penalized',shape=(num_nodes,k,3))
        self.add_output('objective',val=1)
        
        

        # partials
        # define indices
        # self.declare_partials('*', '*', method='fd')
        self.declare_partials('objective','penalized')
        # self.declare_partials('objective','tube_ends_tip')

        
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        penalized = inputs['penalized']
        

        # temp[:,:,1] = (np.tanh(np.outer(np.arange(num_nodes),np.ones(k))-tube_ends_tip[:,1])/2 + 0.5)
        # temp[:,:,2] = (np.tanh(np.outer(np.arange(num_nodes),np.ones(k))-tube_ends_tip[:,2])/2 + 0.5)

        # idx0 = np.where(tube_ends_tip == 0)
        # idx1 = np.where(tube_ends_tip == 1)

        # tube_ends_tip[idx0] = 1
        # tube_ends_tip[idx1] = 0
        # penalize = np.zeros((num_nodes,k,3))
        # penalize = dpsi_ds * tube_ends_tip
                

        outputs['objective'] = np.linalg.norm(penalized)
        
        

        
        

    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        penalized = inputs['penalized']
        ''' Jacobian of partial derivatives for P Pdot matrix.'''

        
        '''Computing Partials'''
        
        sumdpsi = sum(sum(sum(penalized**2))) #+ 1e-10
        dob_dp = ((sumdpsi)**-0.5) * penalized 
        
        
        partials['objective','penalized'] =  dob_dp.flatten()
        



        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    n = 175
    k = 1
    comp.add_output('penalized',val=np.random.random((n,k,3)))
    # comp.add_output('tube_ends_tip',val=np.random.random((k,3)))

    

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = ObjComp(num_nodes=n,k=k)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    prob.model.list_outputs()
    
    
