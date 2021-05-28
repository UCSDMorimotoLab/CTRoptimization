import numpy as np
from openmdao.api import ExplicitComponent
import math


class InterpolationkpComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=3, types=int)
         
    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('straight_ends_hyperbolic',shape=(num_nodes,k,3))
        self.add_input('straight_ends_tip',shape=(k,3))
        self.add_input('kappa',shape=(1,3))


        # outputs
        self.add_output('straight_ends',shape=(num_nodes,k,3))
        
        # partials
        
        col_indices_b = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2])).flatten())\
                             + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        row_indices_b = np.outer(np.arange(num_nodes*k*3),np.ones(3)).flatten()
        self.declare_partials('straight_ends','straight_ends_hyperbolic', rows = row_indices_b,cols=col_indices_b.flatten())
        self.declare_partials('straight_ends', 'straight_ends_tip')
        self.declare_partials('straight_ends', 'kappa')

    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        straight_ends_hyperbolic = inputs['straight_ends_hyperbolic']
        straight_ends_tip = inputs['straight_ends_tip']
        kappa = inputs['kappa']
        interpolation_idx = np.floor(straight_ends_tip).astype(int)
        interpolation_val = straight_ends_tip - np.floor(straight_ends_tip)
        self.interpolate_idx = interpolation_idx
        self.interpolate_val = interpolation_val 
        straight_ends_hyperbolic[interpolation_idx[:,0],:,0] = interpolation_val[:,0] * kappa[:,0]
        straight_ends_hyperbolic[interpolation_idx[:,1],:,1] = interpolation_val[:,1] * kappa[:,1]
        straight_ends_hyperbolic[interpolation_idx[:,2],:,2] = interpolation_val[:,2] * kappa[:,2]
        
        outputs['straight_ends'] = straight_ends_hyperbolic        


    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        interpolation_idx = self.interpolate_idx
        interpolation_val = self.interpolate_val
        """partials Jacobian of partial derivatives."""

        Ps_pkp = np.zeros((num_nodes,k,3,3))
        Ps_pkp[interpolation_idx[:,0],:,0,0] = interpolation_val[:,0]
        Ps_pkp[interpolation_idx[:,1],:,1,1] = interpolation_val[:,1]
        Ps_pkp[interpolation_idx[:,2],:,2,2] = interpolation_val[:,2]
        partials['straight_ends','kappa'][:] = Ps_pkp.reshape((num_nodes*k*3,3))

        
        Pe_pb = np.zeros((num_nodes*k*3,k*3))
        k_ = np.arange(k)
        r_idx0 = interpolation_idx[:,0] * k * 3 + k_ * 3
        c_idx0 = k_*3
        r_idx1 = interpolation_idx[:,1] * k * 3 + 1 + k_ * 3
        c_idx1 = k_*3+1
        r_idx2 = interpolation_idx[:,2] * k * 3 + 2 + k_ * 3
        c_idx2 = k_*3+2
        Pe_pb[:k*3,:k*3] = np.identity(k*3)


        partials['straight_ends','straight_ends_tip'][:] = Pe_pb
        
        Pt_pb = np.zeros((num_nodes,k,3,3))
        Pt_pb[:,:,0,0] = 1
        Pt_pb[:,:,1,1] = 1
        Pt_pb[:,:,2,2] = 1
        Pt_pb[interpolation_idx[:,0],:,0,0] = 0
        Pt_pb[interpolation_idx[:,1],:,1,1] = 0
        Pt_pb[interpolation_idx[:,2],:,2,2] = 0
        
        partials['straight_ends','straight_ends_hyperbolic'][:] = Pt_pb.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n = 17
    k = 4
    idx = int(n/3) 
    hyper = np.zeros((n,k,3))
    hyper[:idx,:,:] = 1
    hyper = np.random.random((n,k,3))
    comp.add_output('straight_ends_hyperbolic',val = hyper )
    beta_init = np.zeros((k,3))
    beta_init[:,0] = 3*n/4
    beta_init[:,1] = n/2
    beta_init[:,2] = n/8
    beta_init = np.random.random((k,3))
    comp.add_output('straight_ends_tip', val=beta_init)
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = InterpolationkpComp(num_nodes=n,k=k)
    group.add_subsystem('interpolationknComp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
  


