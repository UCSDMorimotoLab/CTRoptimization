import numpy as np
from openmdao.api import ExplicitComponent



class ObjsComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=2, types=int)
        self.options.declare('num_nodes', default=3, types=int)
        self.options.declare('zeta')
        self.options.declare('rho')
        self.options.declare('eps_r')
        self.options.declare('eps_p')
        self.options.declare('lag')
        self.options.declare('eps_e')
        self.options.declare('norm1')
        self.options.declare('norm2')
        self.options.declare('norm3')
        self.options.declare('norm4')
        self.options.declare('norm5')
        
        
        
        
        

    '''This class is defining K tensor'''
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        zeta = self.options['zeta']
        #Inputs
        self.add_input('obj1',shape=(k,1))
        self.add_input('targetnorm',shape=(k,1))
        self.add_input('equ_deploylength')
        self.add_input('locnorm')
        self.add_input('rotnorm')
        # outputs
        self.add_output('objs')


        # partials
        
        self.declare_partials('objs', 'obj1')
        self.declare_partials('objs', 'rotnorm')
        self.declare_partials('objs', 'targetnorm')
        self.declare_partials('objs', 'equ_deploylength')
        self.declare_partials('objs', 'locnorm')
        
        



        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        zeta = self.options['zeta']
        rho = self.options['rho']
        eps_r = self.options['eps_r']
        eps_p = self.options['eps_p']
        lag = self.options['lag']
        eps_e = self.options['eps_e']
        norm1 = self.options['norm1']
        norm2 = self.options['norm2']
        norm3 = self.options['norm3']
        norm4 = self.options['norm4']
        norm5 = self.options['norm5']
        obj1 = inputs['obj1']
        equ_deploylength = inputs['equ_deploylength']
        locnorm = inputs['locnorm']
        rotnorm = inputs['rotnorm']
        targetnorm = inputs['targetnorm']

        magnitude = np.sum(zeta * obj1 / norm1)\
                    + eps_e * equ_deploylength / norm2 \
                        + np.sum(0.5 * rho * targetnorm**2 / (norm3**2)) \
                            + np.sum(lag * targetnorm/(norm3)) \
                                + eps_p * locnorm/(norm4) \
                                    + eps_r * rotnorm/(norm5) \
        
        
    
        outputs['objs'] = magnitude.squeeze()



    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""

        k = self.options['k']
        num_nodes = self.options['num_nodes']
        tube_nbr = self.options['tube_nbr']
        zeta = self.options['zeta']
        rho = self.options['rho']
        eps_e = self.options['eps_e']
        eps_r = self.options['eps_r']
        eps_p = self.options['eps_p']
        norm1 = self.options['norm1']
        norm2 = self.options['norm2']
        norm3 = self.options['norm3']
        norm4 = self.options['norm4']
        norm5 = self.options['norm5']
        lag = self.options['lag']
        targetnorm = inputs['targetnorm']

        partials['objs','obj1'][:] = (zeta/norm1).T
        partials['objs','targetnorm'][:] = (rho*targetnorm/(norm3**2) + lag/(norm3)).T
        partials['objs','equ_deploylength'][:] = eps_e/ norm2
        partials['objs','locnorm'][:] = eps_p/norm4
        partials['objs','rotnorm'][:] = eps_r/norm5


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n = 100
    k = 10
    gamma1 = 2
    zeta = 1
    gamma3 = 4
    gamma4 = 3
    comp = IndepVarComp()
   
    
    rho=np.random.random((k,1))
    comp.add_output('obj1', val =np.random.random((k,1)))
    # comp.add_output('sumdistance', val = 10.9)
    comp.add_output('targetnorm', val = np.random.random((k,1)))
    comp.add_output('locnorm', val = 8.8)
    comp.add_output('relativeang', val = 8.8)
    comp.add_output('equ_deploylength', val = 8.8)
    
    


    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = ObjsComp(k=k,num_nodes=n,rho = rho,zeta = np.ones((k,1)),eps_r = gamma3,eps_p=gamma4,lag=rho,norm1 = 1,norm2=0.5,norm3=0.74,norm4=1.99,norm5=0.77,eps_e=zeta )
    group.add_subsystem('testcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    