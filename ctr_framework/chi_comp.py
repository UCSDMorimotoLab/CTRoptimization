import numpy as np
from openmdao.api import ExplicitComponent


class ChiComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('num_t', default=2, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        num_t = self.options['num_t']
        
        

        #Inputs

        self.add_input('tube_section_length',shape=(1,3))
        self.add_input('gamma',shape=(num_nodes,k,3))
        self.add_input('straight_ends', shape=(num_nodes,k,3))
        self.add_input('d2')
        self.add_input('d4')
        self.add_input('d6')
        self.add_input('kappa_eq',shape=(num_nodes,k))
        

        # outputs
        self.add_output('chi_eq',shape=(num_nodes,k,num_t,3))
        self.add_output('chi',shape=(num_nodes,k,num_t,3))



        # partials

        
        row_indices = np.arange(num_nodes*k*num_t*3)
        col_indices = np.outer(np.ones(num_nodes*k*num_t),np.array([0,1,2])).flatten()+np.outer(np.arange(0,num_nodes*k*3,3),np.ones((num_t*3))).flatten()
        row_indices_e = np.arange(num_nodes*k*num_t*3)
        col_indices_e = np.outer(np.arange(num_nodes*k),np.ones((num_t*3))).flatten()
        
        
        
        # print()
        self.declare_partials('chi', 'd2')
        self.declare_partials('chi', 'd4')
        self.declare_partials('chi', 'd6')
        self.declare_partials('chi', 'tube_section_length')
        self.declare_partials('chi', 'straight_ends', rows=row_indices.flatten(), cols=col_indices.flatten())
        self.declare_partials('chi', 'kappa_eq')
        

        self.declare_partials('chi_eq', 'd2')
        self.declare_partials('chi_eq', 'd4')
        self.declare_partials('chi_eq', 'd6')
        self.declare_partials('chi_eq', 'straight_ends')
        self.declare_partials('chi_eq', 'tube_section_length')
        self.declare_partials('chi_eq', 'gamma', rows=row_indices, cols=col_indices)
        self.declare_partials('chi_eq', 'kappa_eq', rows=row_indices_e.flatten(), cols=col_indices_e.flatten())
        
        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        num_t = self.options['num_t']
        d2 = inputs['d2']
        d4 = inputs['d4']
        d6 = inputs['d6']
        tube_section_length = inputs['tube_section_length']
        straight_ends = inputs['straight_ends']
        gamma = inputs['gamma']
        kappa_eq = inputs['kappa_eq']
        link_length = (tube_section_length[:,0]/num_nodes)
        t = [0,np.pi]
        t_tmp = np.reshape(np.tile(t,(num_nodes,k)),(num_nodes,k,num_t))
        self.t_tmp = t_tmp 
       
    
        chi = np.zeros((num_nodes,k,num_t,3))
        chi_eq = np.zeros((num_nodes,k,num_t,3))
        chi[:,:,:,0] = link_length * straight_ends[:,:,np.newaxis,0] * (d2/2)*np.sin(t_tmp-np.pi/2)+1
        chi[:,:,:,1] = link_length * straight_ends[:,:,np.newaxis,1] * (d4/2)*np.sin(t_tmp-np.pi/2)+1
        chi[:,:,:,2] = link_length * straight_ends[:,:,np.newaxis,2] * (d6/2)*np.sin(t_tmp-np.pi/2)+1
        chi_eq[:,:,:,0] = link_length * kappa_eq[:,:,np.newaxis] * (d2/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,0]-np.pi/2)+1
        chi_eq[:,:,:,1] = link_length * kappa_eq[:,:,np.newaxis] * (d4/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,1]-np.pi/2)+1
        chi_eq[:,:,:,2] = link_length * kappa_eq[:,:,np.newaxis] * (d6/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,2]-np.pi/2)+1

        outputs['chi'] = chi
        outputs['chi_eq'] = chi_eq
        
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        num_t = self.options['num_t']
        d2 = inputs['d2']
        d4 = inputs['d4']
        d6 = inputs['d6']
        tube_section_length = inputs['tube_section_length']
        straight_ends = inputs['straight_ends']
        gamma = inputs['gamma']
        kappa_eq = inputs['kappa_eq']
        link_length = (tube_section_length[:,0]/num_nodes)
        t_tmp = self.t_tmp        

        Pc_pd2 = np.zeros((num_nodes,k,num_t,3))
        Pc_pd2[:,:,:,0] = link_length * straight_ends[:,:,np.newaxis,0] * (1/2)*np.sin(t_tmp-np.pi/2)
        Pc_pd4 = np.zeros((num_nodes,k,num_t,3))
        Pc_pd4[:,:,:,1] = link_length * straight_ends[:,:,np.newaxis,1] * (1/2)*np.sin(t_tmp-np.pi/2)
        Pc_pd6 = np.zeros((num_nodes,k,num_t,3))
        Pc_pd6[:,:,:,2] = link_length * straight_ends[:,:,np.newaxis,2] * (1/2)*np.sin(t_tmp-np.pi/2)
        Pc_ptl = np.zeros((num_nodes,k,num_t,3,3))
        Pc_ptl[:,:,:,0,0] = straight_ends[:,:,np.newaxis,0] * (d2/2)*np.sin(t_tmp-np.pi/2)/num_nodes
        Pc_ptl[:,:,:,1,0] = straight_ends[:,:,np.newaxis,1] * (d4/2)*np.sin(t_tmp-np.pi/2)/num_nodes
        Pc_ptl[:,:,:,2,0] = straight_ends[:,:,np.newaxis,2] * (d6/2)*np.sin(t_tmp-np.pi/2)/num_nodes

        Pc_ps = np.zeros((num_nodes,k,num_t,3))
        Pc_ps[:,:,:,0] = link_length * (d2/2)*np.sin(t_tmp-np.pi/2)
        Pc_ps[:,:,:,1] = link_length * (d4/2)*np.sin(t_tmp-np.pi/2)
        Pc_ps[:,:,:,2] = link_length * (d6/2)*np.sin(t_tmp-np.pi/2)

        


        partials['chi','d2'][:] =  Pc_pd2.reshape(-1,1)
        partials['chi','d4'][:] =  Pc_pd4.reshape(-1,1)
        partials['chi','d6'][:] =  Pc_pd6.reshape(-1,1)
        partials['chi','tube_section_length'][:] =  Pc_ptl.reshape(-1,3)
        partials['chi','straight_ends'][:] =  Pc_ps.flatten()
        
        Pce_peq = np.zeros((num_nodes,k,num_t,3))
        Pce_peq[:,:,:,0] = link_length *  (d2/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,0]-np.pi/2)
        Pce_peq[:,:,:,1] = link_length * (d4/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,1]-np.pi/2)
        Pce_peq[:,:,:,2] = link_length * (d6/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,2]-np.pi/2)

        Pce_paeq = np.zeros((num_nodes,k,num_t,3))
        Pce_paeq[:,:,:,0] = link_length * kappa_eq[:,:,np.newaxis] * (d2/2)*np.cos(t_tmp+gamma[:,:,np.newaxis,0]-np.pi/2)
        Pce_paeq[:,:,:,1] = link_length * kappa_eq[:,:,np.newaxis] * (d4/2)*np.cos(t_tmp+gamma[:,:,np.newaxis,1]-np.pi/2)
        Pce_paeq[:,:,:,2] = link_length * kappa_eq[:,:,np.newaxis] * (d6/2)*np.cos(t_tmp+gamma[:,:,np.newaxis,2]-np.pi/2)

        Pce_pd2 = np.zeros((num_nodes,k,num_t,3))
        Pce_pd2[:,:,:,0] = link_length * kappa_eq[:,:,np.newaxis] * (1/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,0]-np.pi/2)
        Pce_pd4 = np.zeros((num_nodes,k,num_t,3))
        Pce_pd4[:,:,:,1] = link_length * kappa_eq[:,:,np.newaxis] * (1/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,1]-np.pi/2)
        Pce_pd6 = np.zeros((num_nodes,k,num_t,3))
        Pce_pd6[:,:,:,2] = link_length * kappa_eq[:,:,np.newaxis] * (1/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,2]-np.pi/2)
        Pce_ptl = np.zeros((num_nodes,k,num_t,3,3))
        Pce_ptl[:,:,:,0,0] = kappa_eq[:,:,np.newaxis] * (d2/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,0]-np.pi/2) / num_nodes
        Pce_ptl[:,:,:,1,0] = kappa_eq[:,:,np.newaxis] * (d4/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,1]-np.pi/2) / num_nodes
        Pce_ptl[:,:,:,2,0] = kappa_eq[:,:,np.newaxis] * (d6/2)*np.sin(t_tmp+gamma[:,:,np.newaxis,2]-np.pi/2) / num_nodes


        


        partials['chi_eq','kappa_eq'][:] =  Pce_peq.flatten()
        partials['chi_eq','gamma'][:] =  Pce_paeq.flatten()
        partials['chi_eq','d2'][:] =  Pce_pd2.reshape(-1,1)
        partials['chi_eq','d4'][:] =  Pce_pd4.reshape(-1,1)
        partials['chi_eq','d6'][:] =  Pce_pd6.reshape(-1,1)
        partials['chi_eq','tube_section_length'][:] =  Pce_ptl.reshape(-1,3)
        # partials['chi_eq','psi'][:] =  Pce_pg.flatten()
        
        

        



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 20
    k = 3
    comp = IndepVarComp()
    t_ends = np.zeros((n,k,3))
    s_ends = np.ones((n,k,3))
    t_ends[:39,:] = 1
    s_ends[:n-1,:] = 0
    # t_ends = np.random.random((n,k,3))
    # s_ends = np.random.random((n,k,3))
    comp.add_output('d2', val=1)
    comp.add_output('d4', val=1)
    comp.add_output('d6', val=1)
    comp.add_output('tube_section_length', val=[1,150,150])
    comp.add_output('gamma', val=np.ones((n,k,3)))
    comp.add_output('straight_ends', val=s_ends)
    

    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ChiComp(num_nodes=n,k=k)
    group.add_subsystem('Chiequilcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    