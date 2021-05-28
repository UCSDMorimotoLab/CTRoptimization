from __future__ import division
from six.moves import range

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from openmdao.api import ImplicitComponent


class UComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=50, types=int)
        self.options.declare('k', default=50, types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        # Inputs
        self.add_input('u2',shape=(num_nodes,k,3,1))
        self.add_input('u3',shape=(num_nodes,k,3,3))
        # Outputs
        self.add_output('u',shape=(num_nodes,k,3,1))
        


        self.declare_partials('u', 'u2')
        row_indices = np.outer(np.arange(num_nodes*k*3),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        col_indices_sk = np.arange(num_nodes*k*3*3).flatten()
        self.declare_partials('u', 'u',rows=row_indices,cols=col_indices.flatten())
        self.declare_partials('u','u3',rows=row_indices,cols=col_indices_sk.flatten())
    def apply_nonlinear(self, inputs, outputs, residuals):
    
        u2 = inputs['u2']
        u3 = inputs['u3']
        u = outputs['u']
        residuals['u']= u3 @ u - u2

    def solve_nonlinear(self, inputs, outputs):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        u2 = inputs['u2']
        u3 = inputs['u3']

        for i in range(3):
            u3[:,:,i,i] = 1/u3[:,:,i,i]
            
        outputs['u'] = u3 @ u2
        
    def linearize(self, inputs, outputs, jacobian):
        num_nodes = self.options['num_nodes']
        k= self.options['k']
        u = outputs['u']
        u3 = inputs['u3']
        jacobian['u', 'u2'][:] = -np.identity(num_nodes*k*3)

        Pu_psk = np.zeros((num_nodes,k,9))
        Pu_psk[:,:,0] = np.reshape(u[:,:,0,:],(num_nodes,k))
        Pu_psk[:,:,1] = np.reshape(u[:,:,1,:],(num_nodes,k))
        Pu_psk[:,:,2] = np.reshape(u[:,:,2,:],(num_nodes,k))
        Pu_psk[:,:,3] = np.reshape(u[:,:,0,:],(num_nodes,k))
        Pu_psk[:,:,4] = np.reshape(u[:,:,1,:],(num_nodes,k))
        Pu_psk[:,:,5] = np.reshape(u[:,:,2,:],(num_nodes,k))
        Pu_psk[:,:,6] = np.reshape(u[:,:,0,:],(num_nodes,k))
        Pu_psk[:,:,7] = np.reshape(u[:,:,1,:],(num_nodes,k))
        Pu_psk[:,:,8] = np.reshape(u[:,:,2,:],(num_nodes,k))
        jacobian['u', 'u3'][:] = Pu_psk.flatten()

        Pu_pu = np.zeros((num_nodes,k,9))
        Pu_pu[:,:,0] = u3[:,:,0,0]
        Pu_pu[:,:,1] = u3[:,:,0,1]
        Pu_pu[:,:,2] = u3[:,:,0,2]  
        Pu_pu[:,:,3] = u3[:,:,1,0]
        Pu_pu[:,:,4] = u3[:,:,1,1]
        Pu_pu[:,:,5] = u3[:,:,1,2]
        Pu_pu[:,:,6] = u3[:,:,2,0]
        Pu_pu[:,:,7] = u3[:,:,2,1]
        Pu_pu[:,:,8] = u3[:,:,2,2]
        jacobian['u', 'u'][:] = Pu_pu.flatten()
        self.jac = np.reshape(Pu_pu,(num_nodes,k,3,3))
         

    def solve_linear(self, u_outputs, u_residuals, mode):
        jac = np.array(self.jac)
            
        if mode == 'fwd':
            u_outputs['u'] = jac @ u_residuals['u']
        else:
            u_residuals['u'] = jac @ u_outputs['u']
            
            
if __name__ == '__main__':

  from openmdao.api import Problem, Group
  from openmdao.api import IndepVarComp


  group = Group()

  comp = IndepVarComp()
  num_nodes = 50
  k = 5
  
  
  comp.add_output('u2',val=np.random.random((num_nodes,k,3,1)))
  comp.add_output('u3',val=np.random.random((num_nodes,k,3,3)))
  group.add_subsystem('Inputcomp', comp, promotes=['*'])



  comp = UComp(num_nodes=num_nodes,k=k)

  group.add_subsystem('ucomp', comp, promotes=['*'])


  prob = Problem()
  prob.model = group
  prob.setup()
  prob.run_model()
#   prob.model.list_outputs()
#   prob.check_partials(compact_print=False)
  prob.check_partials(compact_print=True)