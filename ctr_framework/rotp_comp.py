import numpy as np
from openmdao.api import ExplicitComponent


class RotpComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)
        self.options.declare('base')

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        

        #Inputs
        self.add_input('p', shape=(num_nodes,k,3,1))
        self.add_input('rot',shape=(3,3))
        self.add_input('loc',shape=(3,1))
        # outputs
        self.add_output('rot_p',shape=(num_nodes,k,3,1))
        
        
        row_indices_p = np.outer(np.arange(num_nodes*k*3),np.ones(3)).flatten()
        col_indices_p = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2])).flatten()) \
                            + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        
        row_indices_K = np.outer(np.arange(num_nodes*k*3),np.ones(3)).flatten()
        col_indices_K = np.tile(np.arange(3*3),num_nodes*k).flatten()
        self.declare_partials('rot_p', 'p',rows=row_indices_p,cols=col_indices_p.flatten())
        self.declare_partials('rot_p', 'rot',rows=row_indices_K,cols=col_indices_K)
        col_indices_l = np.outer(np.ones(num_nodes*k),np.outer(np.ones(1),np.array([0,1,2])).flatten()).flatten()
        row_indices_l = np.arange(num_nodes*k*3).flatten()       
        self.declare_partials('rot_p', 'loc',rows=row_indices_l.flatten(),cols=col_indices_l)
       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        p = inputs['p']
        rot = inputs['rot']
        loc = inputs['loc']
        base = self.options['base']

        T = np.zeros((4,4))
        T[:3,:3] = rot
        T[:3,3] = (loc + base).squeeze()
        T[3,3] = 1
    
        p_h = np.zeros((num_nodes,k,4,1))
        p_h[:,:,:3,:] = p
        p_h[:,:,3,:] = 1

        rot_p = T @ p_h
        outputs['rot_p'] = rot_p[:,:,:3,:]
        
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        p = inputs['p']
        rot = inputs['rot']
        
        '''Computing Partials'''
        pd_pp = np.zeros((num_nodes,k,3,3))
        pd_pp[:,:,:,:] = rot
        
        p = np.reshape(p,(num_nodes,k,3))
        pd_pt = np.zeros((num_nodes,k,3,3))
        pd_pt[:,:,0,:] = p
        pd_pt[:,:,1,:] = p
        pd_pt[:,:,2,:] = p


        partials['rot_p','p'][:]= pd_pp.flatten()
        partials['rot_p','rot'][:]= pd_pt.flatten()
        partials['rot_p','loc'][:]= 1

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=20
    k=3
    comp = IndepVarComp()
    comp.add_output('p', val=np.random.random((n,k,3,1)))
    comp.add_output('rot', val = np.identity(3))
    comp.add_output('loc', val = np.random.random((3,1)))
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    base = np.random.random((3,1))
    comp = RotpComp(num_nodes=n,k=k,base=base)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
