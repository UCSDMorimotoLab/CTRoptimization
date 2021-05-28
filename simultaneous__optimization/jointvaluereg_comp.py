import numpy as np
from openmdao.api import ExplicitComponent


class JointvalueregComp(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes',default=3, types=int)

        

    def setup(self):
        #Inputs

        num_nodes = self.options['num_nodes']
        k = self.options['k']
        self.add_input('alpha',shape=(k,3))
        self.add_input('beta',shape=(k,3))
        # self.add_input('kappa',shape=(k,3))
        self.add_output('regularized_jointvalue')


        
        

        # partials
        # define indices
        self.declare_partials('regularized_jointvalue','alpha')
        self.declare_partials('regularized_jointvalue','beta')
        # self.declare_partials('regularized_jointvalue','kappa')

        
        
    def compute(self,inputs,outputs):
        
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        alpha = inputs['alpha']
        beta = inputs['beta']
        # kappa = inputs['kappa']
        


        # temp[:,:,1] = (np.tanh(np.outer(np.arange(num_nodes),np.ones(k))-tube_ends_tip[:,1])/2 + 0.5)
        # temp[:,:,2] = (np.tanh(np.outer(np.arange(num_nodes),np.ones(k))-tube_ends_tip[:,2])/2 + 0.5)

        # idx0 = np.where(tube_ends_tip == 0)
        # idx1 = np.where(tube_ends_tip == 1)

        # tube_ends_tip[idx0] = 1
        # tube_ends_tip[idx1] = 0
        # penalize = np.zeros((num_nodes,k,3))
        # penalize = dpsi_ds * tube_ends_tip
        scalar1 = 1e-2
        scalar2 = 1e-2
        outputs['regularized_jointvalue'] = scalar1*np.linalg.norm(alpha) + scalar2*np.linalg.norm(beta) #+ scalar1*np.linalg.norm(kappa)
        
        

        
        

    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        alpha = inputs['alpha']
        beta = inputs['beta']
        # kappa = inputs['kappa']

        ''' Jacobian of partial derivatives for P Pdot matrix.'''

        
        '''Computing Partials'''
        scalar1 = 1e-2
        scalar2 = 1e-2
        sumdalpha = sum(sum(alpha**2))
        drj_da = ((sumdalpha)**-0.5)* alpha * scalar1
        sumdbeta = sum(sum(beta**2))
        drj_db = ((sumdbeta)**-0.5)* beta * scalar2
        # sumdkappa = sum(sum(kappa**2))
        # drj_dk = ((sumdkappa)**-0.5)* kappa * scalar1
        
        partials['regularized_jointvalue','alpha'] =  drj_da.flatten()
        partials['regularized_jointvalue','beta'] =  drj_db.flatten()
        # partials['regularized_jointvalue','kappa'] =  drj_dk.flatten()
        



        



  




if __name__ == '__main__':
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp

    group = Group()
    
    comp = IndepVarComp()

  
    
    n = 5
    k = 10
    comp.add_output('alpha',val=np.random.random((k,3))*10)
    comp.add_output('beta',val=np.random.random((k,3)))

    

  
    group.add_subsystem('comp1', comp, promotes = ['*'])
    
    
    comp = JointvalueregComp(k=k)
    group.add_subsystem('comp2', comp, promotes = ['*'])
   
    prob = Problem()
    prob.model = group    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    prob.model.list_outputs()
    
    
