import numpy as np
from openmdao.api import ExplicitComponent

"The RHS of psi double dot"

class BaseangleComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('tube_nbr', default=3, types=int)
        self.options.declare('k', default=3, types=int)
        self.options.declare('num_nodes', default=4, types=int)
        self.options.declare('rotx_init')
        self.options.declare('roty_init')
        self.options.declare('rotz_init')

        

    
    def setup(self):
        num_nodes = self.options['num_nodes']
        k = self.options['k']

        #Inputs
        self.add_input('rotx')
        self.add_input('roty')
        self.add_input('rotz')
        # outputs
        self.add_output('rot',shape=(3,3))
        
        
        self.declare_partials('rot', 'rotx')
        self.declare_partials('rot', 'roty')
        self.declare_partials('rot', 'rotz')
       
        
    def compute(self,inputs,outputs):

        k = self.options['k']
        num_nodes= self.options['num_nodes']
        rotx_init = self.options['rotx_init']
        roty_init = self.options['roty_init']
        rotz_init = self.options['rotz_init']

        thetax = inputs['rotx'] + rotx_init
        thetay = inputs['roty'] + roty_init
        thetaz = inputs['rotz'] + rotz_init

        Rotx = np.asarray([[1,0,0],[0,np.cos(thetax),-np.sin(thetax)],[0,np.sin(thetax),np.cos(thetax)]])
        Roty = np.asarray([[np.cos(thetay),0,np.sin(thetay)],[0,1,0],[-np.sin(thetay),0,np.cos(thetay)]])
        Rotz = np.asarray([[np.cos(thetaz),-np.sin(thetaz),0],[np.sin(thetaz),np.cos(thetaz),0],[0,0,1]])

        self.thetax = thetax
        self.thetay = thetay
        self.thetaz = thetaz
        outputs['rot'] = Rotx @ Roty @ Rotz


    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_nodes = self.options['num_nodes']
        k = self.options['k']
        rotx = inputs['rotx']
        roty = inputs['roty']
        rotz = inputs['rotz']
        thetax = self.thetax
        thetay = self.thetay
        thetaz = self.thetaz
        '''Computing Partials'''
        
        Pr_px = np.zeros((3,3))
        Pr_px[1,0] = np.cos(thetax) * np.sin(thetay) * np.cos(thetaz) - np.sin(thetax) * np.sin(thetaz)
        Pr_px[1,1] = -np.cos(thetax) * np.sin(thetay) * np.sin(thetaz) -np.sin(thetax) * np.cos(thetaz)
        Pr_px[1,2] = -np.cos(thetax) * np.cos(thetay)
        Pr_px[2,0] = np.sin(thetax) * np.sin(thetay) * np.cos(thetaz) + np.cos(thetax) * np.sin(thetaz)
        Pr_px[2,1] = -np.sin(thetax) * np.sin(thetay) * np.sin(thetaz) + np.cos(thetax) * np.cos(thetaz)
        Pr_px[2,2] = -np.cos(thetay) * np.sin(thetax)

        Pr_py = np.zeros((3,3))
        Pr_py[0,0] = -np.sin(thetay) * np.cos(thetaz)
        Pr_py[0,1] = np.sin(thetay) * np.sin(thetaz)
        Pr_py[0,2] = np.cos(thetay)
        Pr_py[1,0] = np.sin(thetax) * np.cos(thetay) * np.cos(thetaz)
        Pr_py[1,1] = -np.sin(thetax) * np.cos(thetay) * np.sin(thetaz)
        Pr_py[1,2] = np.sin(thetax) * np.sin(thetay)
        Pr_py[2,0] = -np.cos(thetax) * np.cos(thetay) * np.cos(thetaz) 
        Pr_py[2,1] = np.cos(thetax) * np.cos(thetay) * np.sin(thetaz)
        Pr_py[2,2] = -np.cos(thetax) * np.sin(thetay)  

        Pr_pz = np.zeros((3,3))
        Pr_pz[0,0] = -np.sin(thetaz) * np.cos(thetay)
        Pr_pz[0,1] = -np.cos(thetay) * np.cos(thetaz)
        Pr_pz[1,0] = -np.sin(thetax) * np.sin(thetay) * np.sin(thetaz) + np.cos(thetax) * np.cos(thetaz)
        Pr_pz[1,1] = -np.sin(thetax) * np.sin(thetay) * np.cos(thetaz) - np.cos(thetax) * np.sin(thetaz)
        Pr_pz[2,0] = np.cos(thetax) * np.sin(thetay) * np.sin(thetaz) + np.sin(thetax) * np.cos(thetaz)
        Pr_pz[2,1] = np.cos(thetax) * np.sin(thetay) * np.cos(thetaz) - np.sin(thetax) * np.sin(thetaz)
        

        partials['rot','rotx'][:]= Pr_px.reshape(-1,1)
        partials['rot','roty'][:]= Pr_py.reshape(-1,1)
        partials['rot','rotz'][:]= Pr_pz.reshape(-1,1)


if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    n=2
    k=1
    rotx = -0.1
    roty = -0.1
    rotz = 0.2
    comp = IndepVarComp()
    comp.add_output('p', val=np.random.random((n,k,3,1)))
    comp.add_output('rotx', val = 3.7 )
    comp.add_output('roty', val= -0.1 )
    comp.add_output('rotz', val= -0.2 )
    

    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = BaseangleComp(num_nodes=n,k=k,rotx_init=rotx,roty_init=roty,rotz_init=rotz)
    group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    print('p',prob['p'])
    print('rot',prob['rot'])
    prob.model.list_outputs()

    # prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)