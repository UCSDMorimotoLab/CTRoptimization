import numpy as np
from openmdao.api import ExplicitComponent


class StrainComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('num_t', default=3, types=int)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        num_t = self.options['num_t']
        
        

        #Inputs
        
        self.add_input('chi', shape=(num_nodes,k,num_t,3))
        self.add_input('chi_eq',shape=(num_nodes,k,num_t,3))
        # outputs 
        self.add_output('strain',shape=(num_nodes,k,num_t,3))
        



        # partials

        
        row_indices = np.arange(num_nodes*k*num_t*3)
        col_indices = np.arange(num_nodes*k*num_t*3)
        
        
        
        
        # print()
        self.declare_partials('strain', 'chi',rows=row_indices,cols=col_indices)
        self.declare_partials('strain', 'chi_eq',rows=row_indices,cols=col_indices)
        

        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        chi = inputs['chi']
        chi_eq = inputs['chi_eq']
        
        strain = (chi - chi_eq)/chi



        outputs['strain'] = strain
        
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        chi = inputs['chi']
        chi_eq = inputs['chi_eq']
        
        # Peq_pu = np.zeros((num_nodes,k,3))
        # sumdpsi = np.sum(u**2,axis=2) #+ 1e-10
        # Peq_pu[:,:,0] = (((sumdpsi)**-0.5) * u[:,:,0]).squeeze()
        # Peq_pu[:,:,1] = (((sumdpsi)**-0.5) * u[:,:,1]).squeeze()
        # Peq_pu[:,:,2] = (((sumdpsi)**-0.5) * u[:,:,2]).squeeze()

        partials['strain','chi'][:] = (chi_eq * chi**-2).flatten() 
        partials['strain','chi_eq'][:] = (-chi**-1).flatten()
        
        

        



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 3
    k = 2
    t=2
    comp = IndepVarComp()
    
    # t_ends = np.random.random((n,k,3))
    u = np.random.random((n,k,t,3))*100
    comp.add_output('chi', val=u)
    comp.add_output('chi_eq', val=np.random.rand(n,k,t,3))
    
    

    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = StrainComp(num_nodes=n,k=k,num_t=t)
    group.add_subsystem('Kappaequilcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    