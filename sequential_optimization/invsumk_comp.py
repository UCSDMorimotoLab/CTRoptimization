from __future__ import division
from six.moves import range

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from openmdao.api import ImplicitComponent


class InvsumkComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=50, types=int)
        self.options.declare('k', default=50, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        # Inputs
        self.add_input('sumkm',shape=(num_nodes,k,3,3))
        self.add_input('K',shape=(num_nodes,k,3,3))
        # Outputs
        self.add_output('K_s',shape=(num_nodes,k,3,3))
        

        
        row_indices = np.arange(num_nodes*k*3*3) 
        col_indices = np.arange(num_nodes*k*3*3)
        self.declare_partials('K_s', 'K_s', rows= row_indices, cols= col_indices)
        self.declare_partials('K_s', 'sumkm',rows= row_indices, cols= col_indices)
    
        self.declare_partials('K_s', 'K')
        

    def apply_nonlinear(self, inputs, outputs, residuals):
    
        sumkm = inputs['sumkm']
        K = inputs['K']
        K_s = outputs['K_s']
        residuals['K_s']= sumkm * K_s - K

    def solve_nonlinear(self, inputs, outputs):
        num_nodes = self.options['num_nodes']

        K = inputs['K']
        sumkm = inputs['sumkm']
        
        outputs['K_s'] = K * np.linalg.pinv(sumkm)
        
    def linearize(self, inputs, outputs, jacobian):
        num_nodes = self.options['num_nodes']
        k= self.options['k']
        K_s = outputs['K_s']
        K = inputs['K']
        sumkm = inputs['sumkm']

        jacobian['K_s', 'K'][:] = -np.identity(num_nodes*k*3*3)

        Pu_psk = np.zeros((num_nodes,k,9))
        Pu_psk[:,:,0] = K_s[:,:,0,0]
        Pu_psk[:,:,1] = K_s[:,:,0,1]
        Pu_psk[:,:,2] = K_s[:,:,0,2]
        Pu_psk[:,:,3] = K_s[:,:,1,0]
        Pu_psk[:,:,4] = K_s[:,:,1,1]
        Pu_psk[:,:,5] = K_s[:,:,1,2]
        Pu_psk[:,:,6] = K_s[:,:,2,0]
        Pu_psk[:,:,7] = K_s[:,:,2,1]
        Pu_psk[:,:,8] = K_s[:,:,2,2]
        jacobian['K_s', 'sumkm'][:] = Pu_psk.flatten()

        Pu_pu = np.zeros((num_nodes,k,9))
        Pu_pu[:,:,0] = sumkm[:,:,0,0]
        Pu_pu[:,:,1] = sumkm[:,:,0,1]
        Pu_pu[:,:,2] = sumkm[:,:,0,2]
        Pu_pu[:,:,3] = sumkm[:,:,1,0]
        Pu_pu[:,:,4] = sumkm[:,:,1,1]
        Pu_pu[:,:,5] = sumkm[:,:,1,2]
        Pu_pu[:,:,6] = sumkm[:,:,2,0]
        Pu_pu[:,:,7] = sumkm[:,:,2,1]
        Pu_pu[:,:,8] = sumkm[:,:,2,2]
        jacobian['K_s', 'K_s'][:] = Pu_pu.flatten()
        self.inv_jac = np.linalg.pinv(np.reshape(Pu_pu,(num_nodes,k,3,3)))
        


    def solve_linear(self, K_s_outputs, K_s_residuals, mode):
        
        if mode == 'fwd':
            K_s_outputs['K_s'] = self.inv_jac * K_s_residuals['K_s']
        else:
            K_s_residuals['K_s'] = self.inv_jac *  K_s_residuals['K_s']
         
            
if __name__ == '__main__':

  from openmdao.api import Problem, Group
  from openmdao.api import IndepVarComp


  group = Group()

  comp = IndepVarComp()
  num_nodes = 175
  k = 3

  
  comp.add_output('K',val=np.random.random((num_nodes,k,3,3)))
  comp.add_output('sumkm',val=np.random.random((num_nodes,k,3,3)))
  group.add_subsystem('Inputcomp', comp, promotes=['*'])



  comp = InvsumkComp(num_nodes=num_nodes,k=k)

  group.add_subsystem('ucomp', comp, promotes=['*'])


  prob = Problem()
  prob.model = group
  prob.setup()
  prob.run_model()
#   prob.model.list_outputs()
#   prob.check_partials(compact_print=False)
  prob.check_partials(compact_print=True)