import numpy as np
from openmdao.api import ExplicitComponent


class MaxcurvatureComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('strain', default=0.08, types=float)
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        strain = self.options['strain']
        
        
        

        #Inputs
        self.add_input('kappa', shape=(1,3))
        self.add_input('d2')
        self.add_input('d4')
        self.add_input('d6')

        # outputs
        # self.add_output('curvatureconstraints',shape=(num_nodes,k,6))
        
        self.add_output('curvatureconstraints',shape=(k,6))
        # self.add_output('straight_ends_tip',shape=(k,3))



        # partials

        row_indices = np.arange(num_nodes*k*3)
        col_indices = np.outer(np.arange(num_nodes*k),np.ones(3)).flatten()
        
        self.declare_partials('curvatureconstraints', 'd2')
        self.declare_partials('curvatureconstraints', 'd4')
        self.declare_partials('curvatureconstraints', 'd6')
        self.declare_partials('curvatureconstraints', 'kappa')
        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        strain = self.options['strain']
        
        d2 = inputs['d2']
        d4 = inputs['d4']
        d6 = inputs['d6']
        kappa = inputs['kappa']
        
        #
        diameters = np.zeros((1,3))
        diameters[:,0] = d2
        diameters[:,1] = d4
        diameters[:,2] = d6
        kp_const = np.zeros((k,6))
        kp_const[:,:3] = (2*strain/diameters) - kappa
        kp_const[:,3:] = (2*strain/(diameters*(1+strain))) - kappa
        

        outputs['curvatureconstraints'] = kp_const
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        strain = self.options['strain']
        d2 = inputs['d2']
        d4 = inputs['d4']
        d6 = inputs['d6']
        
        Pt_pd2 = np.zeros((k,6))
        Pt_pd4 = np.zeros((k,6))
        Pt_pd6 = np.zeros((k,6))
        Pt_pkp = np.zeros((k,6,3))

        Pt_pd2[:,0] =  -2*strain/d2**2
        Pt_pd2[:,3] =  -2*strain/(d2**2*(strain + 1))
        Pt_pd4[:,1] =  -2*strain/d4**2
        Pt_pd4[:,4] =  -2*strain/(d4**2*(strain + 1))
        Pt_pd6[:,2] =  -2*strain/d6**2
        Pt_pd6[:,5] =  -2*strain/(d6**2*(strain + 1))

        Pt_pkp[:,[0,3],0] = -1
        Pt_pkp[:,[1,4],1] = -1
        Pt_pkp[:,[2,5],2] = -1
        

        partials['curvatureconstraints','d2'][:] = Pt_pd2.reshape(-1,1)
        partials['curvatureconstraints','d4'][:] = Pt_pd4.reshape(-1,1) 
        partials['curvatureconstraints','d6'][:] = Pt_pd6.reshape(-1,1) 
        partials['curvatureconstraints','kappa'][:] =  Pt_pkp.reshape((k*6,3))
        

        



if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 355
    k = 3
    comp = IndepVarComp()
    
    comp.add_output('d2', val=2)
    comp.add_output('d4', val=10)
    comp.add_output('d6', val=70)
    comp.add_output('kappa', val=np.random.random((1,3)))
    

    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = MaxcurvatureComp(num_nodes=n,k=k)
    group.add_subsystem('MaxcurvatureComp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    