import numpy as np
from openmdao.api import ExplicitComponent
import math


class InterpolationkbComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('tube_ends_hyperbolic',shape=(num_nodes,k,3))
        self.add_input('tube_ends_tip',shape=(k,3))


        # outputs
        self.add_output('tube_ends',shape=(num_nodes,k,3))
        


        # partials
        
        col_indices_b = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2])).flatten())\
                             + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        row_indices_b = np.outer(np.arange(num_nodes*k*3),np.ones(3)).flatten()
        self.declare_partials('tube_ends','tube_ends_hyperbolic', rows = row_indices_b,cols=col_indices_b.flatten())
        self.declare_partials('tube_ends', 'tube_ends_tip')


    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_ends_hyperbolic = inputs['tube_ends_hyperbolic']
        tube_ends_tip = inputs['tube_ends_tip']
        
        interpolation_idx = np.floor(tube_ends_tip).astype(int)
        interpolation_val = tube_ends_tip - np.floor(tube_ends_tip)
        

        self.interpolate_idx = interpolation_idx
        self.interpolate_val = interpolation_val
        tube_ends_hyperbolic[interpolation_idx[:,0],:,0] = interpolation_val[:,0]
        tube_ends_hyperbolic[interpolation_idx[:,1],:,1] = interpolation_val[:,1]
        tube_ends_hyperbolic[interpolation_idx[:,2],:,2] = interpolation_val[:,2]
        outputs['tube_ends'] = tube_ends_hyperbolic        


    def compute_partials(self,inputs,partials):
         num_nodes = self.options['num_nodes']
         k = self.options['k']
         interpolation_idx = self.interpolate_idx
         """partials Jacobian of partial derivatives."""
         '''tube_ends'''
         
         Pe_pb = np.zeros((num_nodes*k*3,k*3))
         k_ = np.arange(k)
         r_idx0 = interpolation_idx[:,0] * k * 3 + k_ * 3
         c_idx0 = k_*3
         r_idx1 = interpolation_idx[:,1] * k * 3 + 1 + k_ * 3
         c_idx1 = k_*3+1
         r_idx2 = interpolation_idx[:,2] * k * 3 + 2 + k_ * 3
         c_idx2 = k_*3+2
         Pe_pb[r_idx0,c_idx0] = 1
         Pe_pb[r_idx1,c_idx1] = 1
         Pe_pb[r_idx2,c_idx2] = 1
         

         partials['tube_ends','tube_ends_tip'][:] = Pe_pb
         
         Pt_pb = np.zeros((num_nodes,k,3,3))
         Pt_pb[:,:,0,0] = 1
         Pt_pb[:,:,1,1] = 1
         Pt_pb[:,:,2,2] = 1
         Pt_pb[interpolation_idx[:,0],:,0,0] = 0
         Pt_pb[interpolation_idx[:,1],:,1,1] = 0
         Pt_pb[interpolation_idx[:,2],:,2,2] = 0
         
         partials['tube_ends','tube_ends_hyperbolic'][:] = Pt_pb.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n = 8
    k = 1
    idx = int(n/2) 
    hyper = np.zeros((n,k,3))
    hyper[:idx,:,:] = 1
    comp.add_output('tube_ends_hyperbolic',val = hyper )
    beta_init = np.zeros((k,3))
    beta_init[:,0] = 3*n/4
    beta_init[:,1] = n/2
    beta_init[:,2] = n/8

    comp.add_output('tube_ends_tip', val=beta_init)
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = InterpolationkbComp(num_nodes=n,k=k)
    group.add_subsystem('interpolationknComp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
  


