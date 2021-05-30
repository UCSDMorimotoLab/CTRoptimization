import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class PathpointsComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('p', shape=(num_nodes,k,3,1))
        self.add_input('desptsconstraints', shape=(k,3))

        # outputs
        self.add_output('pathconstraints',shape=(k,3))
        
        row_indices = np.outer(np.arange(k*3),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(3),np.array([0,1,2])).flatten()) + (np.arange(0,k*3,3).reshape(-1,1))
        # row_indices_p = np.outer(np.arange(k*3),np.ones(num_nodes*3)).flatten()
        # row_indices_p = np.arange(num_nodes*k*3).flatten()
        # col_indices_p = np.outer(np.ones(k*num_nodes),np.outer(np.ones(3),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        row_indices_p = np.outer(np.arange(k*3),np.ones(num_nodes)).flatten()
        col_indices_p = np.outer(np.ones(num_nodes*k),np.outer(np.ones(3),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        row_indices_K = np.outer(np.ones(num_nodes),np.arange(k*3)).flatten()
        col_indices_K = np.arange(num_nodes*k*3).flatten()
        self.declare_partials('pathconstraints', 'p')#,rows= row_indices_K , cols=col_indices_K)#,rows=row_indices, cols=col_indices.flatten())#,rows=row_indices,cols=col_indices)
        self.declare_partials('pathconstraints', 'desptsconstraints')
       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        p = inputs['p']
        tip = inputs['desptsconstraints']
        p = np.reshape(p,(num_nodes,k,3))
        # change here
        idx = np.linspace(20,100,k-1,dtype = int, endpoint=False)
        self.idx = idx
        path = np.zeros((k,3))
        path[0,:] = tip[0,:]
        path[1:,:] = tip[1:,:] - p[idx,0,:]
        
        outputs['pathconstraints'] = path

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        p = inputs['p']
        node_idx = self.idx
        node_idx = np.asarray(node_idx)
        '''Computing Partials'''
        pp_pp = np.zeros((k*3,num_nodes*k*3))
        k_idx = np.arange(k)*3
        
        pp_pp[k_idx[1:],(3*node_idx*k).astype(int)] = -1
        pp_pp[k_idx[1:]+1,(3*node_idx*k+1).astype(int)] = -1
        pp_pp[k_idx[1:]+2,(3*node_idx*k+2).astype(int)] = -1
        # k_ = np.arange(0,k*3,3)
        # pd_pp[np.arange(k)*3,(interpolation_idx_r)*k*3+k_] = tmp
        # pd_pp[np.arange(k)*3+1,(interpolation_idx_r)*k*3+k_+1] = tmp
        # pd_pp[np.arange(k)*3+2,(interpolation_idx_r)*k*3+k_+2] = tmp
        # pd_pp[np.arange(k)*3,(interpolation_idx_l)*k*3+k_] = 1-tmp
        # pd_pp[np.arange(k)*3+1,(interpolation_idx_l)*k*3+k_+1] = 1-tmp
        # pd_pp[np.arange(k)*3+2,(interpolation_idx_l)*k*3+k_+2] = 1-tmp
        
        
        pp_pd = np.identity((k*3))
        # pp_pd[:3,:3] = np.identity(3)
        # pd_pt[:,0] = (p[interpolation_idx_r,np.arange(k),0] - p[interpolation_idx_l,np.arange(k),0]).squeeze()
        # pd_pt[:,3] = (p[interpolation_idx_r,np.arange(k),1] - p[interpolation_idx_l,np.arange(k),1]).squeeze()
        # pd_pt[:,6] = (p[interpolation_idx_r,np.arange(k),2] - p[interpolation_idx_l,np.arange(k),2]).squeeze()
        # pd_pt[:,3] = (p[interpolation_idx_r,:,1] - p[interpolation_idx_l,:,1])
        # pd_pt[:,6] = (p[interpolation_idx_r,:,2] - p[interpolation_idx_l,:,2])

        partials['pathconstraints','desptsconstraints'][:] = pp_pd
        partials['pathconstraints','p'][:]= pp_pp

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=81
    k=3
    comp = IndepVarComp()
    comp.add_output('p', val=np.random.random((n,k,3,1)))
    comp.add_output('desptsconstraints', val=np.random.random((k,3)))
        
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = PathpointsComp(num_nodes=n,k=k)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
