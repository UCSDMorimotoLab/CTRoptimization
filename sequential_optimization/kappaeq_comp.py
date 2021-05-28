import numpy as np
from openmdao.api import ExplicitComponent



class KappaeqComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        

        

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        


        #Inputs
        self.add_input('u',shape=(num_nodes,k,3,1))
        

        # outputs
        self.add_output('kappa_eq',shape=(num_nodes,k))
        self.add_output('angle_eq',shape=(num_nodes,k))

        # partials
        col_indices = np.arange(num_nodes*k*3).flatten()
        row_indices = np.outer(np.arange(num_nodes*k),np.ones(3)).flatten() 
        self.declare_partials('kappa_eq', 'u',rows=row_indices.flatten(),cols=col_indices.flatten())
        self.declare_partials('angle_eq', 'u',rows=row_indices,cols=col_indices)
        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        u = inputs['u']
        magnitude = np.linalg.norm(u,axis=2)
        outputs['kappa_eq'] = magnitude.reshape(num_nodes,k)
        u_tmp = u
        epsilon = 1e-6
        outputs['angle_eq'] = np.arctan2(u[:,:,1],u[:,:,0]).reshape(num_nodes,k)


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        u = inputs['u'].reshape(num_nodes,k,3)
        u = u + 1e-6
        sumdpsi = np.sum(u**2,2)
        pt_pt = np.zeros((num_nodes,k,3))
        pt_pt[:,:,0] = (((sumdpsi)**-0.5) * u[:,:,0])
        pt_pt[:,:,1] = (((sumdpsi)**-0.5) * u[:,:,1])
        pt_pt[:,:,2] = (((sumdpsi)**-0.5) * u[:,:,2])
        partials['kappa_eq','u'][:] = pt_pt.flatten()

        Paeq_pu = np.zeros((num_nodes,k,3))
        epsilon = 1e-6
        u[:,:,:] = u[:,:,:] + epsilon
        Paeq_pu[:,:,0] = (-u[:,:,1] + epsilon)/(u[:,:,0]**2 + u[:,:,1]**2+epsilon)
        Paeq_pu[:,:,1] = (u[:,:,0] + epsilon) / (u[:,:,0]**2 + u[:,:,1]**2+epsilon)

        partials['angle_eq','u'][:] = Paeq_pu.flatten()
        
    


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 5
    k = 2
    comp = IndepVarComp()
    u = np.random.random((n,k,3,1))
    u[1:,3:,0,:] = 0

    comp.add_output('u', val = u)
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = KappaeqComp(k=k,num_nodes=n)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    