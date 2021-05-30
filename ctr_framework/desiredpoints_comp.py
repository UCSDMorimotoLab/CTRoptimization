import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class DesiredpointsComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('rot_p', shape=(num_nodes,k,3,1))
        self.add_input('tube_ends_tip',shape=(k,3))

        # outputs
        self.add_output('desptsconstraints',shape=(k,3))
        
        row_indices = np.outer(np.arange(k*3),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(3),np.array([0,1,2])).flatten()) + (np.arange(0,k*3,3).reshape(-1,1))
        self.declare_partials('desptsconstraints', 'rot_p')
        self.declare_partials('desptsconstraints', 'tube_ends_tip',rows=row_indices, cols=col_indices.flatten())

       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        rot_p = inputs['rot_p']
        tube_ends_tip = inputs['tube_ends_tip']
        rot_p = np.reshape(rot_p,(num_nodes,k,3))
        interpolation_idx_r = np.zeros((k,3))
        interpolation_idx_l = np.zeros((k,3))
        interpolation_val = np.zeros((k,3))
        interpolation_idx_r = np.floor(tube_ends_tip[:,0]).astype(int)
        interpolation_idx_l = np.floor(tube_ends_tip[:,0]).astype(int) - 1
        self.interpolation_idx_r = interpolation_idx_r
        self.interpolation_idx_l = interpolation_idx_l
        tmp = np.ones((k))
        tmp = tube_ends_tip[:,0] - np.floor(tube_ends_tip[:,0])
        self.tmp = tmp
        
        
        interpolation_val[:,0] = rot_p[interpolation_idx_l,np.arange(k),0] \
                +  tmp * (rot_p[interpolation_idx_r,np.arange(k),0] - rot_p[interpolation_idx_l,np.arange(k),0])
        interpolation_val[:,1] = rot_p[interpolation_idx_l,np.arange(k),1] \
                +  tmp * (rot_p[interpolation_idx_r,np.arange(k),1] - rot_p[interpolation_idx_l,np.arange(k),1])
        interpolation_val[:,2] = rot_p[interpolation_idx_l,np.arange(k),2] \
                +  tmp * (rot_p[interpolation_idx_r,np.arange(k),2] - rot_p[interpolation_idx_l,np.arange(k),2])
        
        
        outputs['desptsconstraints'] = interpolation_val


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        rot_p = inputs['rot_p']
        interpolation_idx_r = self.interpolation_idx_r
        interpolation_idx_l = self.interpolation_idx_l
        tmp = self.tmp
        '''Computing Partials'''
        pd_pp = np.zeros((k*3,num_nodes*k*3))
        k_ = np.arange(0,k*3,3)
        pd_pp[np.arange(k)*3,(interpolation_idx_r)*k*3+k_] = tmp
        pd_pp[np.arange(k)*3+1,(interpolation_idx_r)*k*3+k_+1] = tmp
        pd_pp[np.arange(k)*3+2,(interpolation_idx_r)*k*3+k_+2] = tmp
        pd_pp[np.arange(k)*3,(interpolation_idx_l)*k*3+k_] = 1-tmp
        pd_pp[np.arange(k)*3+1,(interpolation_idx_l)*k*3+k_+1] = 1-tmp
        pd_pp[np.arange(k)*3+2,(interpolation_idx_l)*k*3+k_+2] = 1-tmp
        
        pd_pt = np.zeros((k,9))
        pd_pt[:,0] = (rot_p[interpolation_idx_r,np.arange(k),0] - rot_p[interpolation_idx_l,np.arange(k),0]).squeeze()
        pd_pt[:,3] = (rot_p[interpolation_idx_r,np.arange(k),1] - rot_p[interpolation_idx_l,np.arange(k),1]).squeeze()
        pd_pt[:,6] = (rot_p[interpolation_idx_r,np.arange(k),2] - rot_p[interpolation_idx_l,np.arange(k),2]).squeeze()
        
        partials['desptsconstraints','tube_ends_tip'][:] = pd_pt.flatten()
        partials['desptsconstraints','rot_p'][:]= np.reshape(pd_pp,(k*3,num_nodes*k*3))

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=175
    k=2
    comp = IndepVarComp()
    comp.add_output('rot_p', val=np.random.random((n,k,3,1)))
    comp.add_output('tube_ends_tip', val=([1.3,3,3],[2.1,3.1,3.1]))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = DesiredpointsComp(num_nodes=n,k=k)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
