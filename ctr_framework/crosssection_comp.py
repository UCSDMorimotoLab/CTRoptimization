import numpy as np
from openmdao.api import ExplicitComponent


class CrosssectionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        
        
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']


        #Inputs
        self.add_input('d2')
        self.add_input('d4')
        self.add_input('d6')
        self.add_input('tube_ends',shape=(num_nodes,k,3))


        # outputs
        self.add_output('cross_section', shape=(num_nodes,k))


        # partials
        row_indices_st = np.outer(np.arange(0,num_nodes*k),np.ones(3))
        col_indices_st = np.arange(num_nodes*k*3)
        self.declare_partials('cross_section', 'd2')
        self.declare_partials('cross_section', 'd4')
        self.declare_partials('cross_section', 'd6')
        self.declare_partials('cross_section', 'tube_ends',rows=row_indices_st.flatten(),cols=col_indices_st.flatten())

        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        d2 = inputs['d2']
        d4 = inputs['d4']
        d6 = inputs['d6']
        tube_ends = inputs['tube_ends']
        
        
        
        # cross section radii
        tube1 = np.zeros((num_nodes,k))
        tube2 = np.zeros((num_nodes,k))
        tube3 = np.zeros((num_nodes,k))
        cross_section = np.zeros((num_nodes,k))
        tube1 = (tube_ends[:,:,0] - tube_ends[:,:,1])
        tube2 = (tube_ends[:,:,1] - tube_ends[:,:,2])
        tube3 = tube_ends[:,:,2]
        self.tube1 = tube1
        self.tube2 = tube2
        self.tube3 = tube3
        cross_section = tube1  * d2/2  + tube2 * d4/2 + tube3 * d6/2
        self.cross_section = cross_section
        
        outputs['cross_section'] = cross_section
        
        
        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        d2 = inputs['d2']
        d4 = inputs['d4']
        d6 = inputs['d6']
        tube_ends = inputs['tube_ends']
        
        tube1 = self.tube1
        tube2 = self.tube2
        tube3 = self.tube3
        cross_section = self.cross_section
        

        # partial d2
        Pc_pd2 = np.zeros((num_nodes,k))
        Pc_pd2 = tube1/2
        
        
        # partial d4
        Pc_pd4 = np.zeros((num_nodes,k))
        Pc_pd4 = tube2/2
        
        # partial d6
        Pc_pd6 = np.zeros((num_nodes,k))
        Pc_pd6 = tube3/2
        
        # partial tube_ends
        Pc_pt = np.zeros((num_nodes,k,3))
        Pc_pt[:,:,0] = d2/2
        Pc_pt[:,:,1] = -d2/2 + d4/2
        Pc_pt[:,:,2] = -d4/2 + d6/2
        
        
    
        partials['cross_section','tube_ends'][:] = Pc_pt.flatten()
        partials['cross_section','d2'][:] = np.reshape(Pc_pd2,(num_nodes*k,1))
        partials['cross_section','d4'][:] = np.reshape(Pc_pd4,(num_nodes*k,1))
        partials['cross_section','d6'][:] = np.reshape(Pc_pd6,(num_nodes*k,1))


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 200
    k = 2
    comp = IndepVarComp()
    # comp.add_output('tube_section_length', val=0.1 * np.ones((2,3)))
    # comp.add_output('beta', val=0.1 * np.ones((2,3)))
    

    
    comp.add_output('d2', val = 2.5)
    comp.add_output('d4', val = 2.7)
    comp.add_output('d6', val = 3)
    t_ends = np.zeros((n,k,3))
    t_ends[:5,:,:] = 1
    
    # comp.add_output('tube_ends', val = t_ends)
    comp.add_output('tube_ends', val = np.random.random((n,k,3)))
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = CrosssectionComp(k=k,num_nodes=n)
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    