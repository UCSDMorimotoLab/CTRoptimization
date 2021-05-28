import numpy as np
from openmdao.api import ExplicitComponent


class EqudplyComp(ExplicitComponent):

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


        #Inputs
        self.add_input('deploy_length',shape=(k,3))
        

        # outputs
        self.add_output('equ_deploylength')


        # partials
        self.declare_partials('equ_deploylength', 'deploy_length')
        
        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        deploy_length = inputs['deploy_length']
        
        w1 = 1
        w2 = 0.1
        w3 = 5
        
        'formulation 4'
        temp = w1 * (deploy_length[:,0] - deploy_length[:,1])**2 + w2 * (deploy_length[:,1] -  deploy_length[:,2])**2
        'formulation 5'
       
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.temp = temp
        # outputs['equ_deploylength'] = np.sum(equ)
        outputs['equ_deploylength'] = np.sum(temp)



    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        deploy_length = inputs['deploy_length']

        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        temp = self.temp
        pe_pd = np.zeros((k,3))
        # pe_pd[:,0] = w1
        # pe_pd[:,1] = w2
        # pe_pd[:,2] = w3
        # 'formulation 3'
        # pe_pd[:,0] = 2 * ((deploy_length[:,0] - deploy_length[:,1])) * w1 + 2 * ((deploy_length[:,0] - deploy_length[:,2])) * w2
        # pe_pd[:,1] = -2 * ((deploy_length[:,0] - deploy_length[:,1])) * w1 
        # pe_pd[:,2] = -2 * ((deploy_length[:,0] - deploy_length[:,2])) * w2
        'formulation 4'
        pe_pd[:,0] = 2 * ((deploy_length[:,0] - deploy_length[:,1])) * w1 
        pe_pd[:,1] = -2 * ((deploy_length[:,0] - deploy_length[:,1])) * w1 + 2 * ((deploy_length[:,1] - deploy_length[:,2])) * w2
        pe_pd[:,2] = -2 * ((deploy_length[:,1] - deploy_length[:,2])) * w2
        'f 5'
        '''pe_pd[:,0] = (2*deploy_length[:,0] - 2*deploy_length[:,1])/deploy_length[:,0]**2 - 2*(deploy_length[:,2]**2 + 
                        (deploy_length[:,0] - deploy_length[:,1])**2 + (-deploy_length[:,2] + deploy_length[:,1])**2)/deploy_length[:,0]**3
        pe_pd[:,1] = (-2*deploy_length[:,0] - 2*deploy_length[:,2] + 4*deploy_length[:,1])/deploy_length[:,0]**2
        pe_pd[:,2] = (4*deploy_length[:,2] - 2*deploy_length[:,1])/deploy_length[:,0]**2'''
        # pe_pd[:,0] = 2 * (deploy_length[:,0]-deploy_length[:,1]) / (30)**2  
        # pe_pd[:,1] = (-2 * (deploy_length[:,0]-deploy_length[:,1]) + 2 * (deploy_length[:,1]-deploy_length[:,2]))/ (30)**2  
        # pe_pd[:,2] = (-2 * (deploy_length[:,1]-deploy_length[:,2]) + 2*deploy_length[:,2])/ (30)**2  
        partials['equ_deploylength','deploy_length'][:] = pe_pd.reshape(1,-1)


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 100
    k = 3
    comp = IndepVarComp()

    

    comp.add_output('deploy_length', val = np.random.rand(k,3))
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = EqudplyComp(k=k,num_nodes=n)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    