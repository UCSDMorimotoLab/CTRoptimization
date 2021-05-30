import numpy as np
from openmdao.api import ExplicitComponent

class Ptequdistant2Comp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_pt', default=3, types=int)
        self.options.declare('pt_')
       
    def setup(self):
        num_pt = self.options['num_pt']
        

        #Inputs
        self.add_input('dis_sum')

        # outputs
        self.add_output('path_obj2')
        self.declare_partials('path_obj2', 'dis_sum')

       
        
    def compute(self,inputs,outputs):

        pt_ = self.options['pt_']
        num_pt = self.options['num_pt']
        dis_sum = inputs['dis_sum']
        
        
        total_sum = np.sum(np.sqrt(np.sum((pt_[:num_pt-1,:] - pt_[1:,:])**2,1)))**2
        self.total_sum = total_sum
        path_obj2 = dis_sum / total_sum
        outputs['path_obj2'] = path_obj2
        

    def compute_partials(self,inputs,partials):
        """ partials Jacobian of partial derivatives."""
        dis_sum = inputs['dis_sum']
        total_sum = self.total_sum
        '''Computing Partials'''

        partials['path_obj2','dis_sum'][:]= 1/total_sum

if __name__ == '__main__':
    
    from openmdao.api import Problem, Group
    
    from openmdao.api import IndepVarComp
    
    group = Group()
    num_pt=50
    pt_ = np.random.rand(num_pt,3)
    comp = IndepVarComp()
    comp.add_output('dis_sum', val=10)
    # comp.add_output('total_sum', val=100)
    
    group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
    
    
    comp = Ptequdistant2Comp(num_pt=num_pt,pt_=pt_)
    group.add_subsystem('startpointcomp', comp, promotes = ['*'])
    
    prob = Problem()
    prob.model = group
    
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=False)
    prob.check_partials(compact_print=True)
    # prob.check_partials(compact_print=False)
