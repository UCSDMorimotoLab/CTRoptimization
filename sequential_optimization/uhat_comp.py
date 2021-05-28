import numpy as np
from openmdao.api import ExplicitComponent


class UhatComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

    '''This class is defining the sin() tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('u', shape=(num_nodes,k,3,1))

        # outputs
        self.add_output('uhat',shape=(num_nodes,k,3,3))
        
        row_indices_S = np.outer(np.arange(0,num_nodes*k*3*3),np.ones(3))
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('uhat', 'u', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
       
        


        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        u = inputs['u']
        u = np.reshape(u,(num_nodes,k,3))
                
        uhat = np.zeros((num_nodes,k,3,3))
        #skew-symmetric matrix
        uhat[:,:,0,1] = -u[:,:,2]
        uhat[:,:,0,2] = u[:,:,1]
        uhat[:,:,1,0] = u[:,:,2]
        uhat[:,:,1,2] = -u[:,:,0]
        uhat[:,:,2,0] = -u[:,:,1]
        uhat[:,:,2,1] = u[:,:,0]

        outputs['uhat'] = uhat


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        u = inputs['u']
        
        
        # partial

        'u'
        Puhat_u = np.zeros((num_nodes,k,9,3))
        Puhat_u[:,:, 5,0] = -1
        Puhat_u[:,:, 7,0] = 1
        
        Puhat_u[:,:, 2,1] = 1
        Puhat_u[:,:, 6,1] = -1
        
        Puhat_u[:,:, 1,2] = -1
        Puhat_u[:,:, 3,2] = 1
        partials['uhat','u'] = Puhat_u.flatten()
        


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=1
    k=2
    comp = IndepVarComp()
    comp.add_output('u', val=np.random.random((n,k,3,1)))
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = UhatComp(num_nodes=n,k=k)
    group.add_subsystem('ucomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
