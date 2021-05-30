import numpy as np
from openmdao.api import ExplicitComponent


class StraightendsComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('a', default=30, types=int)
        
        
    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        #Inputs
        self.add_input('beta', shape=(k,3))
        self.add_input('kappa', shape=(1,3))
        self.add_input('tube_section_straight',shape=(1,3))
        self.add_input('tube_section_length',shape=(1,3))

        # outputs
        self.add_output('straight_ends_hyperbolic',shape=(num_nodes,k,3))
        self.add_output('straight_ends_tip',shape=(k,3))



        # partials

        col_indices_b = np.outer(np.ones(num_nodes),np.arange(3*k)).flatten()
        row_indices_b = np.arange(num_nodes*k*3).flatten()
        self.declare_partials('straight_ends_hyperbolic', 'kappa')
        self.declare_partials('straight_ends_hyperbolic', 'beta',rows=row_indices_b,cols=col_indices_b)
        self.declare_partials('straight_ends_hyperbolic', 'tube_section_straight')
        self.declare_partials('straight_ends_hyperbolic', 'tube_section_length')

        row_indices = np.outer(np.arange(0,k*3),np.ones(3)).flatten()
        col_indices = np.outer(np.ones(k),np.outer(np.ones(3),np.array([0,1,2])).flatten()) \
                            + (np.arange(0,k*3,3).reshape(-1,1))
        self.declare_partials('straight_ends_tip', 'beta',rows = row_indices, cols = col_indices.flatten())
        self.declare_partials('straight_ends_tip', 'tube_section_straight')
        self.declare_partials('straight_ends_tip', 'tube_section_length')
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        a = self.options['a']
        kappa = inputs['kappa']
        beta = inputs['beta']
        tube_section_length = inputs['tube_section_length']
        tube_section_straight = inputs['tube_section_straight']
        t1_idx = []
        t2_idx = []
        t3_idx = []

        deployed_length = tube_section_straight + beta
        for i in range(k):
            if deployed_length[i,0] < 0 :
                deployed_length[i,0] = 0
                t1_idx.append(i)
            if deployed_length[i,1] < 0 :
                deployed_length[i,1] = 0
                t2_idx.append(i)
            if deployed_length[i,2] < 0 :
                deployed_length[i,2] = 0
                t3_idx.append(i)

        straight_ends_tip = np.zeros((k,3))
        link_length = tube_section_length[:,0] / num_nodes
        straight_ends_tip = deployed_length / link_length
        self.t1_idx = np.array(t1_idx)
        self.t2_idx = np.array(t2_idx)
        self.t3_idx = np.array(t3_idx)
        self.straight_ends_tip = straight_ends_tip
        temp = np.zeros((num_nodes,k,3))
        temp[:,:,0] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-straight_ends_tip[:,0]))/2 + 0.5) * kappa[:,0]
        temp[:,:,1] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-straight_ends_tip[:,1]))/2 + 0.5) * kappa[:,1]
        temp[:,:,2] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-straight_ends_tip[:,2]))/2 + 0.5) * kappa[:,2]
        outputs['straight_ends_hyperbolic'] = temp
        outputs['straight_ends_tip'] = straight_ends_tip
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        a = self.options['a']

        beta = inputs['beta']
        kappa = inputs['kappa']
        tube_section_length = inputs['tube_section_length']
        tube_section_straight = inputs['tube_section_straight']
        t1_idx = self.t1_idx
        t2_idx = self.t2_idx
        t3_idx = self.t3_idx
        straight_ends_tip = self.straight_ends_tip

        'tube_section_straight'
        # a = 10
        Pt_pts = np.zeros((num_nodes,k,3,3))
        x = (np.outer(np.arange(1,num_nodes+1),np.ones(k)))
                        
        Pt_pts[:,:,0,0] = -0.5*a*kappa[:,0]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,0] + tube_section_straight[:,0]) \
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pts[:,:,1,1] = -0.5*a*kappa[:,1]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,1] + tube_section_straight[:,1]) \
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pts[:,:,2,2] = -0.5*a*kappa[:,2]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,2] + tube_section_straight[:,2]) \
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pts[:,t1_idx.astype(int),0,0] = 0
        Pt_pts[:,t2_idx.astype(int),1,1] = 0
        Pt_pts[:,t3_idx.astype(int),2,2] = 0
        
         
        'tube_section_length'
        Pt_pt = np.zeros((num_nodes,k,3,3))
        Pt_pt[:,:,0,0] = 0.5*a*kappa[:,0]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,0] + tube_section_straight[:,0])\
                                            /tube_section_length[:,0] + x))**2)*(beta[:,0] + tube_section_straight[:,0])/tube_section_length[:,0]**2
        Pt_pt[:,:,1,0] = 0.5*a*kappa[:,1]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,1] + tube_section_straight[:,1])\
                                            /tube_section_length[:,0] + x))**2)*(beta[:,1] + tube_section_straight[:,1])/tube_section_length[:,0]**2
        Pt_pt[:,:,2,0] = 0.5*a*kappa[:,2]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,2] + tube_section_straight[:,2])\
                                            /tube_section_length[:,0] + x))**2)*(beta[:,2] + tube_section_straight[:,2])/tube_section_length[:,0]**2
        Pt_pt[:,t1_idx.astype(int),0,0] = 0
        Pt_pt[:,t2_idx.astype(int),1,0] = 0
        Pt_pt[:,t3_idx.astype(int),2,0] = 0
        
        'beta'
        Pt_pb = np.zeros((num_nodes,k,3))
        Pt_pb[:,:,0] = -0.5*a*kappa[:,0]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,0] + tube_section_straight[:,0])\
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pb[:,:,1] = -0.5*a*kappa[:,1]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,1] + tube_section_straight[:,1])\
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pb[:,:,2] = -0.5*a*kappa[:,2]*num_nodes*(1 - np.tanh(a*(-num_nodes*(beta[:,2] + tube_section_straight[:,2])\
                                            /tube_section_length[:,0] + x))**2)/tube_section_length[:,0]
        Pt_pb[:,t1_idx.astype(int),0] = 0
        Pt_pb[:,t2_idx.astype(int),1] = 0
        Pt_pb[:,t3_idx.astype(int),2] = 0
        
        'kappa'
        Pt_pk = np.zeros((num_nodes,k,3,3))

        
        Pt_pk[:,:,0,0] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-(tube_section_straight[:,0]+beta[:,0])\
                                            /(tube_section_length[:,0]/num_nodes)))/2 + 0.5)
        Pt_pk[:,:,1,1] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-(tube_section_straight[:,1]+beta[:,1])\
                                            /(tube_section_length[:,0]/num_nodes)))/2 + 0.5)
        Pt_pk[:,:,2,2] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-(tube_section_straight[:,2]+beta[:,2])\
                                            /(tube_section_length[:,0]/num_nodes)))/2 + 0.5)
        Pt_pk[:,:,0,0] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-straight_ends_tip[:,0]))/2 + 0.5) 
        Pt_pk[:,:,1,1] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-straight_ends_tip[:,1]))/2 + 0.5) 
        Pt_pk[:,:,2,2] = (np.tanh(a*(np.outer(np.arange(1,num_nodes+1),np.ones(k))-straight_ends_tip[:,2]))/2 + 0.5) 
    

        partials['straight_ends_hyperbolic','tube_section_straight'][:] =  Pt_pts.reshape((num_nodes*k*3,3))
        partials['straight_ends_hyperbolic','tube_section_length'][:] =  Pt_pt.reshape((num_nodes*k*3,3))
        partials['straight_ends_hyperbolic','beta'][:] =  Pt_pb.flatten()
        partials['straight_ends_hyperbolic','kappa'][:] =  Pt_pk.reshape((num_nodes*k*3,3))

        # beta
        Pe_pb = np.zeros((k,3,3))
        Pe_pb[:, 0,0] = (num_nodes/(tube_section_length[:,0]))
        Pe_pb[:, 1,1] = (num_nodes/(tube_section_length[:,0]))
        Pe_pb[:, 2,2] = (num_nodes/(tube_section_length[:,0]))
        Pe_pb[t1_idx.astype(int),0,0] = 0
        Pe_pb[t2_idx.astype(int),1,1] = 0
        Pe_pb[t3_idx.astype(int),2,2] = 0
        # tube_section_straight
        Pe_pt = np.zeros((k,3,3))
        Pe_pt[:, 0,0] = (num_nodes/(tube_section_length[:,0]))
        Pe_pt[:, 1,1] = (num_nodes/(tube_section_length[:,0]))
        Pe_pt[:, 2,2] = (num_nodes/(tube_section_length[:,0]))
        Pe_pt[t1_idx.astype(int),0,0] = 0
        Pe_pt[t2_idx.astype(int),1,1] = 0
        Pe_pt[t3_idx.astype(int),2,2] = 0
        #tube_section_length
        Pe_pl = np.zeros((k,3,3))
        Pe_pl[:, 0,0] = -num_nodes*(beta[:,0] + tube_section_straight[:,0])/tube_section_length[:,0]**2
        Pe_pl[:, 1,0] = -num_nodes*(beta[:,1] + tube_section_straight[:,1])/tube_section_length[:,0]**2
        Pe_pl[:, 2,0] = -num_nodes*(beta[:,2] + tube_section_straight[:,2])/tube_section_length[:,0]**2
        Pe_pl[t1_idx.astype(int),0,0] = 0
        Pe_pl[t2_idx.astype(int),1,0] = 0
        Pe_pl[t3_idx.astype(int),2,0] = 0

        partials['straight_ends_tip','tube_section_straight'][:] =  Pe_pt.reshape((k*3,3))
        partials['straight_ends_tip','tube_section_length'][:] =  Pe_pl.reshape((k*3,3))
        partials['straight_ends_tip','beta'][:] =  Pe_pb.flatten()
        



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 175
    k = 2
    comp = IndepVarComp()
    kappa_init = np.array([0.031, 0.0231,0.006]).reshape((1,3))
    comp.add_output('kappa', val=kappa_init)
    beta_init = np.zeros((k,3))
    beta_init[:,0] = -55
    beta_init[:,1] = -40
    beta_init[:,2] = -25
    comp.add_output('beta', val=beta_init)
    comp.add_output('tube_section_length', val=[175,120,65])
    comp.add_output('tube_section_straight', val = [160,70,15])
    

    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = StraightendsComp(num_nodes=n,k=k)
    group.add_subsystem('Straightendscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    