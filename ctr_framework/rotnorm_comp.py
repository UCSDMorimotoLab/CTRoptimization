import numpy as np
from openmdao.api import ExplicitComponent


class RotnormComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        
        self.add_input('rotx',shape=(k))
        self.add_input('roty',shape=(k))
        self.add_input('rotz',shape=(k))
        # self.add_input('loc',shape=(2,1))
        # outputs
        self.add_output('rotnorm')
        
        
        self.declare_partials('rotnorm', 'rotx')
        self.declare_partials('rotnorm', 'roty')
        self.declare_partials('rotnorm', 'rotz')
        
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        
        rotx = inputs['rotx']
        roty = inputs['roty']
        rotz = inputs['rotz']
        rotxyz = np.zeros((k,3))
        rotxyz[:,0] = rotx
        rotxyz[:,1] = roty
        rotxyz[:,2] = rotz
        self.rotxyz = rotxyz
        norm = np.linalg.norm(rotxyz)
        # norm = np.sqrt(rotx**2+roty**2)
        outputs['rotnorm'] = norm

        


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        rotx = inputs['rotx']
        roty = inputs['roty']
        rotz = inputs['rotz']
        rotxyz = self.rotxyz
        
         
        

        '''Computing Partials'''
        partials['rotnorm','rotx'][:]= rotx*(np.sum(rotxyz**2)**-0.5)
        partials['rotnorm','roty'][:]= roty*(np.sum(rotxyz**2)**-0.5)
        partials['rotnorm','rotz'][:]= rotz*(np.sum(rotxyz**2)**-0.5)

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=2
    k=2
    comp = IndepVarComp()
    
    comp.add_output('rotx', val = np.random.rand(k))
    comp.add_output('roty', val = np.random.rand(k))
    comp.add_output('rotz', val = np.random.rand(k))
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = RotnormComp(num_nodes=n,k=k)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
