import numpy as np
from openmdao.api import ExplicitComponent


class DpComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k')
        self.options.declare('p_')
        self.options.declare('num_nodes')
        
    
        
    def setup(self):
        num_nodes = self.options['num_nodes']
        p_ = self.options['p_']
        k = self.options['k']
        

        #Inputs
        self.add_input('rot_p', shape=(num_nodes,k,3))
        self.add_input('tube_ends',shape=(num_nodes,k,3))


        # outputs
        self.add_output('normalized_dis',shape=(num_nodes,k,p_.shape[0],3))
        self.add_output('euclidean_dist',shape=(num_nodes,k,p_.shape[0]))


        # partials
        row_indices = np.outer(np.arange(0,num_nodes*k*p_.shape[0]),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(num_nodes*k),np.outer(np.ones(p_.shape[0]),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('euclidean_dist', 'rot_p',rows=row_indices,cols=col_indices.flatten())
        row_indices_n = np.outer(np.arange(0,num_nodes*k*p_.shape[0]*3),np.ones(3)).flatten()
        col_indices_n = np.outer(np.ones(num_nodes*k),np.outer(np.ones(p_.shape[0]*3),np.array([0,1,2])).flatten()) + (np.arange(0,num_nodes*k*3,3).reshape(-1,1))
        self.declare_partials('normalized_dis', 'rot_p',rows=row_indices_n,cols=col_indices_n.flatten())
        self.declare_partials('normalized_dis', 'tube_ends',rows=row_indices_n,cols=col_indices_n.flatten())
        

        
        
    def compute(self,inputs,outputs):

        num_nodes = self.options['num_nodes']
        p_ = self.options['p_']
        k = self.options['k']
        rot_p = inputs['rot_p']
        tube_ends = inputs['tube_ends']

        dis = np.zeros((num_nodes,k,p_.shape[0],3))
        normalized_dis = np.zeros((num_nodes,k,p_.shape[0],3))
        
        p_tmp = np.zeros((num_nodes,k,p_.shape[0],3))
        p_tmp[:,:,:,:] = np.reshape(np.tile(rot_p,p_.shape[0]),(num_nodes,k,p_.shape[0],3))
        dis = p_tmp - p_
        norm_dis = np.linalg.norm(dis,axis=3)
        epsilon = 1e-8
        normalized_dis[:,:,:,0] = dis[:,:,:,0] / (norm_dis+epsilon)
        normalized_dis[:,:,:,1] = dis[:,:,:,1] / (norm_dis+epsilon)
        normalized_dis[:,:,:,2] = dis[:,:,:,2] / (norm_dis+epsilon)
        self.dis = dis
        self.norm_dis = norm_dis
        self.normalized_dis = normalized_dis
        euclidean_dist = np.sum((dis)**2,axis=3)**0.125
        outputs['normalized_dis'] = normalized_dis * tube_ends[:,:,0,np.newaxis,np.newaxis]
        outputs['euclidean_dist'] = euclidean_dist 
        
        
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self
        k = self.options['k']
        rot_p = inputs['rot_p']
        tube_ends = inputs['tube_ends']
        p_ = self.options['p_']
        num_nodes = self.options['num_nodes']
        dis = self.dis
        norm_dis = self.norm_dis
        normalized_dis = self.normalized_dis
        Peu_ppt = np.zeros((num_nodes,k,p_.shape[0],3))
        for i in range(3):
            Peu_ppt[:,:,:,i] = (np.sum((dis)**2,axis=3) **-0.875)*dis[:,:,:,i] * 0.25
        
        Pnd_ppt = np.zeros((num_nodes,k,p_.shape[0],3,3))
                
        Pnd_ppt[:,:,:,0,0] =  (1/norm_dis + dis[:,:,:,0] * -0.5*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,0] )* tube_ends[:,:,0,np.newaxis]
        Pnd_ppt[:,:,:,1,1] =  (1/norm_dis + dis[:,:,:,1] * -0.5*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,1]) * tube_ends[:,:,0,np.newaxis]
        Pnd_ppt[:,:,:,2,2] =  (1/norm_dis + dis[:,:,:,2] * -0.5*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,2] )* tube_ends[:,:,0,np.newaxis]
        Pnd_ppt[:,:,:,0,1] =  -0.5*dis[:,:,:,0]*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,1] * tube_ends[:,:,0,np.newaxis]
        Pnd_ppt[:,:,:,0,2] =  -0.5*dis[:,:,:,0]*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,2]* tube_ends[:,:,0,np.newaxis]
        Pnd_ppt[:,:,:,1,0] =  -0.5*dis[:,:,:,1]*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,0]* tube_ends[:,:,0,np.newaxis]
        Pnd_ppt[:,:,:,1,2] =  -0.5*dis[:,:,:,1]*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,2]* tube_ends[:,:,0,np.newaxis]
        Pnd_ppt[:,:,:,2,0] =  -0.5*dis[:,:,:,2]*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,0]* tube_ends[:,:,0,np.newaxis]
        Pnd_ppt[:,:,:,2,1] =  -0.5*dis[:,:,:,2]*(np.sum(dis**2,3)**-1.5) * 2 * dis[:,:,:,1]* tube_ends[:,:,0,np.newaxis]

        Pnd_pt = np.zeros((num_nodes,k,p_.shape[0],3,3))
        Pnd_pt[:,:,:,:,0] =  normalized_dis
        
        partials['normalized_dis','rot_p'][:] = Pnd_ppt.flatten()
        partials['normalized_dis','tube_ends'][:] = Pnd_pt.flatten()
        partials['euclidean_dist','rot_p'][:] = Peu_ppt.flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    num_nodes = 200
    k=1
    
    comp = IndepVarComp()
    rot_p = np.random.rand(110,3)
    normals = np.random.rand(110,3)


    comp.add_output('rot_p', val = np.random.random((num_nodes,k,3))*10)
    comp.add_output('tube_ends', val = np.random.random((num_nodes,k,3)))

    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = DpComp(num_nodes=num_nodes,k=k,p_=rot_p)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    