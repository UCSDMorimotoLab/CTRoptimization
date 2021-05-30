import numpy as np
from openmdao.api import ExplicitComponent

class Ptequdistant1Comp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_pt', default=3, types=int)
                

    def setup(self):
        num_pt = self.options['num_pt']
        

        #Inputs
        self.add_input('pt', shape=(num_pt,3))

        # outputs
        self.add_output('dis_sum')
        
        self.declare_partials('dis_sum', 'pt')

       
        
    def compute(self,inputs,outputs):

        num_pt = self.options['num_pt']
        pt = inputs['pt']
        
        dis_btw_pt = np.zeros((num_pt-1,3))
        dis_btw_pt = pt[:num_pt-1,:] - pt[1:,:]
        sum_dis = np.sum(np.sum(dis_btw_pt**2,1))

        t = np.sum(np.sqrt(np.sum(dis_btw_pt**2,1)))
        self.dis_btw_pt = dis_btw_pt
        self.t = t
        outputs['dis_sum'] = sum_dis
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        num_pt = self.options['num_pt']
        pt = inputs['pt']
        dis_btw_pt = self.dis_btw_pt
        t = self.t
        '''Computing Partials'''
        ps_pcp = np.zeros((num_pt,3))
        ps_pcp[0,:] = 2 * dis_btw_pt[0]
        ps_pcp[1:num_pt-1,:] = -2 * (dis_btw_pt[:num_pt-2,:]) + 2 * (dis_btw_pt[1:,:])
        ps_pcp[-1,:] = -2 * dis_btw_pt[-1]

        partials['dis_sum','pt'][:]= np.reshape(ps_pcp,(1,num_pt*3))

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    num_pt=150
    pt_ = np.random.rand(150,3)
    comp = IndepVarComp()
    comp.add_output('pt', val=np.random.random((num_pt,3)))
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = Ptequdisant1Comp(num_pt=num_pt)
    group.add_subsystem('startpointcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
