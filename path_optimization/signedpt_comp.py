import numpy as np
from openmdao.api import ExplicitComponent



class SignedptComp(ExplicitComponent):

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
        self.add_input('normalized_dis',shape=(num_pt,p.shape[0],3))
        self.add_input('euclidean_dist',shape=(num_pt,p.shape[0]))
    
        # outputs
        self.add_output('path_obj1')


        # partials
        self.declare_partials('path_obj1', 'normalized_dis')
        self.declare_partials('path_obj1', 'euclidean_dist')
        

        
        
    def compute(self,inputs,outputs):

        num_pt = self.options['num_pt']
        normals = self.options['normals']
        normalized_dis = inputs['normalized_dis']
        euclidean_dist = inputs['euclidean_dist']

        normals_ = np.zeros((num_pt,normals.shape[0],3))
        normals_[np.arange(num_pt),:,:] = normals
        self.normals_ = normals_
        inner_product = np.einsum("ijk,ijk->ij", normalized_dis,normals_)
        self.inner_product = inner_product
        f = (-1*inner_product)/(euclidean_dist)
        path_obj1 = np.sum(np.sum(f))
        
        outputs['path_obj1'] = path_obj1
        
        
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        
        num_pt = self.options['num_pt']
        normalized_dis = inputs['normalized_dis']
        euclidean_dist = inputs['euclidean_dist']
        p = self.options['p_']
        normals_ = self.normals_
        inner_product = self.inner_product

        Pob1_pno = np.zeros((num_pt,p.shape[0],3))
        Pob1_pno[:,:,:] = (-1 * normals_[:,:,:] / euclidean_dist[:,:,np.newaxis])
        
        Pob1_peu = np.zeros((num_pt,p.shape[0]))
        Pob1_peu[:,:] = inner_product / (euclidean_dist)**2
        
        partials['path_obj1','normalized_dis'][:] = Pob1_pno.flatten()
        partials['path_obj1','euclidean_dist'][:] = Pob1_peu.flatten()

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    from mesh import trianglemesh
    group = Group()
    num_pt = 10
    p = np.random.rand(100,3)*10
    normals = np.random.rand(100,3)
    comp = IndepVarComp()
    
    comp.add_output('normalized_dis',val=np.random.random((num_pt,p.shape[0],3)))
    comp.add_output('euclidean_dist',val=np.random.random((num_pt,p.shape[0])))
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = SignedptComp(num_pt=num_pt,p_=p,normals=normals)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    