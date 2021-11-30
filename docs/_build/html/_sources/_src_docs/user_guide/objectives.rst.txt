Objective functions
=================================

The objective function in the optimization includes two main implicit enforced constraints (anatomical constraints and tip position) 
that use the penalty method formulation. Due to the modularity of the framework, adding the new objective term in the optimization
is simple and fast. The user would need to modify the component for the objective by adding a new input.
Below using the ObjsComp (an OpenMDAO component) demonstrates how to add a new objective term in the objective function.



.. code-block:: python
        
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
            # New user-defined objective can be added here as an input 
            self.add_input('new_obj')

            # outputs
            self.add_output('objs')


            # partials
            self.declare_partials('objs', 'obj1')
            self.declare_partials('objs', 'rotnorm')
            self.declare_partials('objs', 'targetnorm')
            self.declare_partials('objs', 'equ_deploylength')
            self.declare_partials('objs', 'locnorm')
            # declare the partials of objective function with respect to new objective term
            self.declare_partials('objs', 'new_obj')

            
            
            
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
            new_obj = inputs['new_obj']

            magnitude = np.sum(zeta * obj1 / norm1)\
                        + eps_e * equ_deploylength / norm2 \
                            + np.sum(0.5 * rho * targetnorm**2 / (norm3**2)) \
                                + np.sum(lag * targetnorm/(norm3)) \
                                    + eps_p * locnorm/(norm4) \
                                        + eps_r * rotnorm/(norm5) \
                                            + new_obj # new term can be added here  
            
            
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
        # compute the partials and give the model analytically
        partials['objs','new_obj'][:] = 1 


        


.. toctree::
  :maxdepth: 2
  :titlesonly:
