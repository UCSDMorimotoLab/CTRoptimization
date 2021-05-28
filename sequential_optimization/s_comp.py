import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class SComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('E', default=80.0, types=float)
        self.options.declare('J', default=80.0, types=float)
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        

    '''This class is defining the sin() tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('psi', shape=(num_nodes,k,3))
        

        # outputs
        self.add_output('S',shape=(num_nodes,k,3,3))
        
        row_indices_S = np.outer(np.arange(0,num_nodes*k*3*3),np.ones(3))
        col_indices_S = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2])).flatten())\
                         + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('S', 'psi', rows=row_indices_S.flatten(), cols=col_indices_S.flatten())
       

        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        psi = inputs['psi']
        
        

        
        
        S = np.zeros((num_nodes,k,3,3))
        s = np.ones((num_nodes,k,3))
        for i in range(tube_nbr):
            for z in range(tube_nbr):
                S[:,:,i,z] = np.sin(psi[:,:,i] - psi[:,:,z])
        
        outputs['S'] = S


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        tube_nbr = self.options['tube_nbr']
        k = self.options['k']
        num_nodes = self.options['num_nodes']
        psi = inputs['psi']
        
        Pk_ps = np.zeros((num_nodes*k,9,3))
        Pk_ps[:, 0,0] = np.zeros(num_nodes*k)
        Pk_ps[:, 1,0] = np.cos(psi[:,:,0]-psi[:,:,1]).flatten()
        Pk_ps[:, 2,0] = np.cos(psi[:,:,0]-psi[:,:,2]).flatten()
        Pk_ps[:, 3,0] = -np.cos(psi[:,:,0]-psi[:,:,1]).flatten()
        Pk_ps[:, 4,0] = np.zeros(num_nodes*k)
        Pk_ps[:, 5,0] = np.zeros(num_nodes*k)
        Pk_ps[:, 6,0] = -np.cos(psi[:,:,0]-psi[:,:,2]).flatten()
        Pk_ps[:, 7,0] = np.zeros(num_nodes*k)
        Pk_ps[:, 8,0] = np.zeros(num_nodes*k)


        Pk_ps[:, 0,1] = np.zeros(num_nodes*k) 
        Pk_ps[:, 1,1] = -np.cos(psi[:,:,0]-psi[:,:,1]).flatten()
        Pk_ps[:, 2,1] = np.zeros(num_nodes*k)
        Pk_ps[:, 3,1] = np.cos(psi[:,:,0]-psi[:,:,1]).flatten()
        Pk_ps[:, 4,1] = np.zeros(num_nodes*k) 
        Pk_ps[:, 5,1] = np.cos(psi[:,:,1]-psi[:,:,2]).flatten()
        Pk_ps[:, 6,1] = np.zeros(num_nodes*k) 
        Pk_ps[:, 7,1] = -np.cos(psi[:,:,1]-psi[:,:,2]).flatten()
        Pk_ps[:, 8,1] = np.zeros(num_nodes*k) 

        Pk_ps[:, 0,2] = np.zeros(num_nodes*k)
        Pk_ps[:, 1,2] = np.zeros(num_nodes*k)
        Pk_ps[:, 2,2] = -np.cos(psi[:,:,0]-psi[:,:,2]).flatten()
        Pk_ps[:, 3,2] = np.zeros(num_nodes*k)
        Pk_ps[:, 4,2] = np.zeros(num_nodes*k)
        Pk_ps[:, 5,2] = -np.cos(psi[:,:,1]-psi[:,:,2]).flatten()
        Pk_ps[:, 6,2] = np.cos(psi[:,:,0]-psi[:,:,2]).flatten()
        Pk_ps[:, 7,2] = np.cos(psi[:,:,1]-psi[:,:,2]).flatten()
        Pk_ps[:, 8,2] = np.zeros(num_nodes*k)

        partials['S','psi'] = Pk_ps.flatten()
        
        


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=175
    k=1
    comp = IndepVarComp()
    comp.add_output('psi', val=np.random.random((n,k,3)))
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = SComp(num_nodes=n,k=k)
    group.add_subsystem('Scomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
