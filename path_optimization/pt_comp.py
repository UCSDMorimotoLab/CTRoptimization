import numpy as np
from openmdao.api import ExplicitComponent


class PtComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('num_pt', default=2, types=int)
        self.options.declare('num_pt', default=3, types=int)
        self.options.declare('p_')
        self.options.declare('normals')
        
    
        
    def setup(self):
        num_pt = self.options['num_pt']
        normals = self.options['normals']
        p = self.options['p_']
        #Inputs
        self.add_input('pt', shape=(num_pt,3))


        # outputs
        self.add_output('normalized_dis',shape=(num_pt,p.shape[0],3))
        self.add_output('euclidean_dist',shape=(num_pt,p.shape[0]))


        # partials
        row_indices = np.outer(np.arange(0,num_pt*p.shape[0]),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(num_pt),np.outer(np.ones(p.shape[0]),np.array([0,1,2])).flatten()) + (np.arange(0,num_pt*3,3).reshape(-1,1))
        self.declare_partials('euclidean_dist', 'pt',rows=row_indices,cols=col_indices.flatten())
        row_indices_n = np.outer(np.arange(0,num_pt*p.shape[0]*3),np.ones(3)).flatten()
        col_indices_n = np.outer(np.ones(num_pt),np.outer(np.ones(p.shape[0]*3),np.array([0,1,2])).flatten()) + (np.arange(0,num_pt*3,3).reshape(-1,1))
        self.declare_partials('normalized_dis', 'pt',rows=row_indices_n,cols=col_indices_n.flatten())
        

        
        
    def compute(self,inputs,outputs):

        num_pt = self.options['num_pt']
        normals = self.options['normals']
        p = self.options['p_']
        pt = inputs['pt']

        dis = np.zeros((num_pt,p.shape[0],3))
        normalized_dis = np.zeros((num_pt,p.shape[0],3))
        
        for i in range(num_pt):
            dis[i,:,:] = pt[i,:] -  p
        norm_dis = np.linalg.norm(dis,axis=2)
        epsilon = 1e-8
        normalized_dis[:,:,0] = dis[:,:,0] / (norm_dis+epsilon)
        normalized_dis[:,:,1] = dis[:,:,1] / (norm_dis+epsilon)
        normalized_dis[:,:,2] = dis[:,:,2] / (norm_dis+epsilon)
        self.dis = dis
        self.norm_dis = norm_dis
        euclidean_dist = np.sqrt(np.sum((dis)**2,axis=2))
        euclidean_dist = norm_dis
    
        outputs['normalized_dis'] = normalized_dis
        outputs['euclidean_dist'] = euclidean_dist
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_pt = self
        pt = inputs['pt']
        # normals = self.options['normals']
        p = self.options['p_']
        num_pt = self.options['num_pt']
        dis = self.dis
        norm_dis = self.norm_dis
        
        # partial p
        Peu_ppt = np.zeros((num_pt,p.shape[0],3))
        Peu_ppt[:,:,0] = (np.sum((dis)**2,axis=2) **-0.5)*dis[:,:,0]
        Peu_ppt[:,:,1] = (np.sum((dis)**2,axis=2) **-0.5)*dis[:,:,1]
        Peu_ppt[:,:,2] = (np.sum((dis)**2,axis=2) **-0.5)*dis[:,:,2]
        
        
        
        Pnd_ppt = np.zeros((num_pt,p.shape[0],3,3))
        Pnd_ppt[:,:,0,0] =  1/norm_dis + dis[:,:,0] * -0.5*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,0]
        Pnd_ppt[:,:,1,1] =  1/norm_dis + dis[:,:,1] * -0.5*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,1]
        Pnd_ppt[:,:,2,2] =  1/norm_dis + dis[:,:,2] * -0.5*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,2]
        Pnd_ppt[:,:,0,1] =  -0.5*dis[:,:,0]*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,1]
        Pnd_ppt[:,:,0,2] =  -0.5*dis[:,:,0]*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,2]
        Pnd_ppt[:,:,1,0] =  -0.5*dis[:,:,1]*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,0]
        Pnd_ppt[:,:,1,2] =  -0.5*dis[:,:,1]*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,2]
        Pnd_ppt[:,:,2,0] =  -0.5*dis[:,:,2]*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,0]
        Pnd_ppt[:,:,2,1] =  -0.5*dis[:,:,2]*(np.sum(dis**2,2)**-1.5) * 2 * dis[:,:,1]



        partials['normalized_dis','pt'][:] = Pnd_ppt.flatten()
        partials['euclidean_dist','pt'][:] = Peu_ppt.flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    num_pt = 10
    
    comp = IndepVarComp()
    # comp.add_output('tube_section_length', val=0.1 * np.ones((2,3)))
    # comp.add_output('beta', val=0.1 * np.ones((2,3)))
    p = np.random.rand(30,3)
    normals = np.random.rand(30,3)


    comp.add_output('pt', val = np.random.random((num_pt,3))*10)
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = PtComp(num_pt=num_pt,p_=p,normals=normals)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    