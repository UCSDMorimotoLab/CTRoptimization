import numpy as np
from openmdao.api import ExplicitComponent




class TuberadiiComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        #self.pcd = o3d.io.read_point_cloud("/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/torsionally_compliant/ctr_optimization/mesh/lumen1.PLY")
        #self.mesh = o3d.io.read_triangle_mesh("/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/torsionally_compliant/ctr_optimization/mesh/lumen1.PLY")
        
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        self.mesh  = trianglemesh(num_nodes,k)


        #Inputs
        self.add_input('', shape=(num_nodes,k,3))
        

        # outputs
        self.add_output('nneighbors', shape=(num_nodes,k))


        # partials

        ''' kb '''
        row_indices_st = np.outer(np.arange(0,num_nodes*k),np.ones(3))
        # col_indices_st = np.outer(np.ones(num_nodes*k),np.outer(np.ones(9),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        col_indices_st = np.arange(num_nodes*k*3)
        self.declare_partials('nneighbors', 'p',rows=row_indices_st.flatten(),cols=col_indices_st.flatten())
        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        mesh = self.mesh
        p = inputs['p']
        
        # find nearest neighbors
        query,normals = mesh.nn(p)
        dis = p-query
        # norm_dis = np.linalg.norm(dis,axis=2)
        # norm_normals = np.linalg.norm(normals,axis=2)
        # normalized_dis = np.zeros((num_nodes,k,3))
        # normalized_dis[:,:,0] = dis[:,:,0] / norm_dis
        # normalized_dis[:,:,1] = dis[:,:,1] / norm_dis
        # normalized_dis[:,:,2] = dis[:,:,2] / norm_dis
        
        # check robot is inside or not
        inner_product = np.einsum("ijk,ijk->ij", dis, normals)
        idx_negative = np.where((inner_product<0))
        idx_negative = np.array(idx_negative)
        epsilon = 1e-8
        self.dis = dis
        self.epsilon = epsilon
        self.idx_negative = idx_negative
        # distance between robot poins and anatomy points
        euclidean_dist = np.zeros((num_nodes,k))
        euclidean_dist = np.sqrt(np.sum((dis)**2,axis=2))
        
        # penalize in cost function
        # inner product > 0
        magnitude = np.zeros((k,3))
        magnitude = 1 / (euclidean_dist+epsilon)
        # inner product < 0
        # penalize = 1/ ((euclidean_dist+epsilon)*1e-8)
        penalize = euclidean_dist*1e+3
        # self.penalize = penalize
        magnitude[idx_negative[0,:],idx_negative[1,:]] = penalize[idx_negative[0,:],idx_negative[1,:]]
        # magnitude[idx_negative[0,:],idx_negative[1,:]] = 1e+3
        magnitude[0,:] = 0   
        outputs['nneighbors'] = magnitude
        # print(magnitude)
        
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        p = inputs['p']
        dis = self.dis
        epsilon = self.epsilon
        idx_negative = self.idx_negative
        Pnn_pp = np.zeros((num_nodes,k,3))
        Pnn_pp[:,:,0] = -((np.sum((dis)**2,axis=2)+epsilon)**-1.5)* dis[:,:,0]
        Pnn_pp[:,:,1] = -((np.sum((dis)**2,axis=2)+epsilon)**-1.5)* dis[:,:,1]
        Pnn_pp[:,:,2] = -((np.sum((dis)**2,axis=2)+epsilon)**-1.5)* dis[:,:,2]
        Pnn_pp[idx_negative[0,:],idx_negative[1,:],0] = 0
        Pnn_pp[idx_negative[0,:],idx_negative[1,:],1] = 0
        Pnn_pp[idx_negative[0,:],idx_negative[1,:],2] = 0
        temp = np.zeros((num_nodes,k,3))
        temp[:,:,0] = ((np.sum((dis)**2,axis=2)+epsilon)**-0.5)* dis[:,:,0] * 1e+3
        temp[:,:,1] = ((np.sum((dis)**2,axis=2)+epsilon)**-0.5)* dis[:,:,1] * 1e+3
        temp[:,:,2] = ((np.sum((dis)**2,axis=2)+epsilon)**-0.5)* dis[:,:,2] * 1e+3
        Pnn_pp[idx_negative[0,:],idx_negative[1,:],0] = temp[idx_negative[0,:],idx_negative[1,:],0]
        Pnn_pp[idx_negative[0,:],idx_negative[1,:],1] = temp[idx_negative[0,:],idx_negative[1,:],1]
        Pnn_pp[idx_negative[0,:],idx_negative[1,:],2] = temp[idx_negative[0,:],idx_negative[1,:],2]
        Pnn_pp[0,:,:] = 0
        partials['nneighbors','p'][:] = Pnn_pp.flatten()


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 10
    k = 2
    comp = IndepVarComp()
    # comp.add_output('tube_section_length', val=0.1 * np.ones((2,3)))
    # comp.add_output('beta', val=0.1 * np.ones((2,3)))
    

    comp.add_output('p', val = np.random.random((n,k,3)))
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = KdtreeComp(k=k,num_nodes=n)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    