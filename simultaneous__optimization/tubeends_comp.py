import numpy as np
from openmdao.api import ExplicitComponent
import math


class TubeendsComp(ExplicitComponent):

    def initialize(self):
        
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('a', default = 30, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('tube_section_length',shape=(1,3))
        self.add_input('beta',shape=(k,3))


        # outputs
        self.add_output('tube_ends_hyperbolic',shape=(num_nodes,k,3))
        self.add_output('tube_ends_tip',shape=(k,3))
        self.add_output('deploy_length',shape=(k,3))

        


        # partials
        
        self.declare_partials('tube_ends_hyperbolic','tube_section_length')
        col_indices_b = np.outer(np.ones(num_nodes),np.arange(3*k)).flatten()
        row_indices_b = np.arange(num_nodes*k*3).flatten()
        self.declare_partials('tube_ends_hyperbolic', 'beta', rows= row_indices_b,cols= col_indices_b)

        row_indices = np.outer(np.arange(0,k*3),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(3),np.array([0,1,2])).flatten()) + (np.arange(0,k*3,3).reshape(-1,1))
        self.declare_partials('tube_ends_tip','tube_section_length')
        self.declare_partials('tube_ends_tip', 'beta' , rows=row_indices,cols=col_indices.flatten())

        row_indices_d = np.arange(k*3)
        col_indices_d = np.outer(np.ones(k),np.array([0,1,2])).flatten()
        self.declare_partials('deploy_length','tube_section_length',rows=row_indices_d,cols=col_indices_d)
        self.declare_partials('deploy_length', 'beta')
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        a = self.options['a']
        tube_section_length = inputs['tube_section_length']
        beta = inputs['beta']

        # compute the deployed length and apply zero stiffness
        deployed_length = np.zeros((k,3))
        deployed_length = tube_section_length + beta
        link_length = tube_section_length[:,0] / num_nodes
        tube_ends = (deployed_length / link_length) 
       
        
        temp = np.zeros((num_nodes,k,3))
        
        temp[:,:,0] = 1-(np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends[:,0]))/2 + 0.5)
        temp[:,:,1] = 1-(np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends[:,1]))/2 + 0.5)
        temp[:,:,2] = 1-(np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-tube_ends[:,2]))/2 + 0.5)
        
        outputs['tube_ends_hyperbolic'] = temp
        outputs['tube_ends_tip'] = tube_ends
        outputs['deploy_length'] = deployed_length
        


    def compute_partials(self,inputs,partials):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        a = self.options['a']
        tube_section_length = inputs['tube_section_length']
        beta = inputs['beta']
        deployed_length = tube_section_length + beta
        link_length = tube_section_length[:,0] / num_nodes
        tube_ends = (deployed_length / link_length) 
        
        Pe_pt = np.zeros((k,3,3))
        Pe_pt[:, 0,0] = num_nodes/tube_section_length[:,0] - num_nodes*(beta[:,0] + tube_section_length[:,0])/tube_section_length[:,0]**2
        Pe_pt[:, 1,0] = -num_nodes*(beta[:,1] + tube_section_length[:,1])/tube_section_length[:,0]**2
        Pe_pt[:, 2,0] = -num_nodes*(beta[:,2] + tube_section_length[:,2])/tube_section_length[:,0]**2
        Pe_pt[:, 1,1] = (num_nodes/(tube_section_length[:,0]))
        Pe_pt[:, 2,2] = (num_nodes/(tube_section_length[:,0]))
    #  beta
        
        Pe_pb = np.zeros((k,3,3))
        Pe_pb[:, 0,0] = (num_nodes/(tube_section_length[:,0]))
        Pe_pb[:, 1,1] = (num_nodes/(tube_section_length[:,0]))
        Pe_pb[:, 2,2] = (num_nodes/(tube_section_length[:,0]))
        partials['tube_ends_tip','tube_section_length'][:] = Pe_pt.reshape((k*3,3))
        partials['tube_ends_tip','beta'][:] = Pe_pb.flatten()
        
    
        Pt_pt = np.zeros((num_nodes,k,3,3))
        x = (np.outer(np.arange(1,num_nodes+1),np.ones(k)))
        
        Pt_pt[:,:,0,0] = -0.5*a*(1 - np.tanh(a*(-num_nodes*(beta[:,0] + tube_section_length[:,0])\
                                    /tube_section_length[:,0] + x))**2)*(-num_nodes/tube_section_length[:,0] + num_nodes*(beta[:,0] + tube_section_length[:,0])\
                                    /tube_section_length[:,0]**2)
        Pt_pt[:,:,1,0] = -0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,1] + tube_section_length[:,1])\
                                    /tube_section_length[:,0] + x))**2)*(beta[:,1] + tube_section_length[:,1])/tube_section_length[:,0]**2
        Pt_pt[:,:,2,0] = -0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,2] + tube_section_length[:,2])\
                                    /tube_section_length[:,0] + x))**2)*(beta[:,2] + tube_section_length[:,2])/tube_section_length[:,0]**2
        Pt_pt[:,:,1,1] = 0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,1] + tube_section_length[:,1])\
                                    /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pt[:,:,2,2] = 0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,2] + tube_section_length[:,2])\
                                    /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        
        Pt_pb = np.zeros((num_nodes,k,3))
        Pt_pb[:,:,0] = 0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,0] + tube_section_length[:,0])\
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pb[:,:,1] = 0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,1] + tube_section_length[:,1])\
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pb[:,:,2] = 0.5*a*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,2] + tube_section_length[:,2])\
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        
        partials['tube_ends_hyperbolic','tube_section_length'][:] = Pt_pt.reshape((num_nodes*k*3,3))
        partials['tube_ends_hyperbolic','beta'][:] = Pt_pb.flatten()

        'deploy length'
        partials['deploy_length','beta'][:] = np.identity(k*3)
        partials['deploy_length','tube_section_length'][:] = np.ones(k*3)
        


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    
    comp = IndepVarComp()
    n = 175
    k = 5
    comp.add_output('tube_section_length',val = [175,120,65])
    beta_init = np.zeros((k,3))
    beta_init[:,0] = -20.5
    beta_init[:,1] = -40.7
    beta_init[:,2] = -25.9

    comp.add_output('beta', val=beta_init)
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = TubeendsComp(num_nodes=n,k=k)
    group.add_subsystem('TubeendsComp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    