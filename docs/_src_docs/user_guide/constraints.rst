Constraints
=========================

CTR design optimization framework provides the freedom to the user to add any task-specific constraints. The modular model approach
reduce the significant anmount of effort to build custimized optimization model. The user can easily build a component in OpenMDAO
and added them as a constraints in the model. The kinematics constraints are the constraints in order to solve the kinematics models
and also includes some physical constraints, such as tube clearance, and wall thickness, which user can defined upon their needs.

Kinematics constraints
-----------------------
Here shows how the example of the kinematics and tube geomerty constraints are added in the optimization.

.. code-block:: python

    '''Constraints'''
        bccomp = BcComp(num_nodes=num_nodes,k=k)
        diametercomp = DiameterComp()
        tubeclearancecomp = TubeclearanceComp()
        tubestraightcomp = TubestraightComp()
        baseplanarcomp = BaseplanarComp(num_nodes=num_nodes,k=k,equ_paras=equ_paras)
        deployedlenghtcomp = DeployedlengthComp(k=k)
        betacomp = BetaComp(k=k)

        self.add_subsystem('BetaComp', betacomp, promotes=['*'])
        self.add_subsystem('BcComp', bccomp, promotes=['*'])
        self.add_subsystem('Baseplanarcomp', baseplanarcomp, promotes=['*'])
        self.add_subsystem('DeployedlengthComp', deployedlenghtcomp, promotes=['*'])
        self.add_subsystem('TubestraightComp', tubestraightcomp, promotes=['*'])
        self.add_subsystem('DiameterComp', diametercomp, promotes=['*'])
        self.add_subsystem('TubeclearanceComp', tubeclearancecomp, promotes=['*'])

    
        self.add_constraint('torsionconstraint', equals=0.)
        self.add_constraint('baseconstraints', lower=0)
        self.add_constraint('deployedlength12constraint', lower=1)
        self.add_constraint('deployedlength23constraint', lower=1)
        self.add_constraint('beta12constraint', upper=-1)
        self.add_constraint('beta23constraint', upper=-1)
        self.add_constraint('diameterconstraint',lower= 0.1)
        self.add_constraint('tubeclearanceconstraint',lower= 0.1,upper=0.16)
        self.add_constraint('tubestraightconstraint',lower= 0)
        


Task-specific constraints
-------------------------

Adding OpenMDAO component
~~~~~~~~~~~~~~~~~~~~~~~~~

An OpenMDAO component needs to be created in order to add a constraint to an optimization.
In this example, the user would like to add a constraint on the robot tip orientation.  

.. code-block:: python

    import numpy as np
    from openmdao.api import ExplicitComponent

    class TiporientationComp(ExplicitComponent):

        def initialize(self):
            self.options.declare('tube_nbr', default=3, types=int)
            self.options.declare('k', default=3, types=int)
            self.options.declare('num_nodes', default=4, types=int)
            self.options.declare('tar_vector')
            

        
        def setup(self):
            num_nodes = self.options['num_nodes']
            k = self.options['k']

            #Inputs
            self.add_input('desptsconstraints',shape=(k,3))

            # outputs
            self.add_output('tiporientation',shape=(k))
            
            row_indices = np.outer(np.arange(k),np.ones(3)).flatten()
            col_indices = np.arange(k*3)
            
            self.declare_partials('tiporientation', 'desptsconstraints',rows=row_indices,cols=col_indices)
            

        
            
        def compute(self,inputs,outputs):

            k = self.options['k']
            tar_vector = self.options['tar_vector']
            desptsconstraints = inputs['desptsconstraints']
            
            dot = (desptsconstraints - tar_vector[:,0]) @  (tar_vector[:,1] - tar_vector[:,0])
            
            outputs['tiporientation'] = dot


        def compute_partials(self,inputs,partials):
            """ partials Jacobian of partial derivatives."""
            num_nodes = self.options['num_nodes']
            k = self.options['k']
            tar_vector = self.options['tar_vector']
            
            '''Computing Partials'''
            pd_pp = np.zeros((k,3))
            pd_pp[:,:] = (tar_vector[:,1] - tar_vector[:,0]).T

            partials['tiporientation','desptsconstraints'][:] = pd_pp.flatten()

    if __name__ == '__main__':
        
        from openmdao.api import Problem, Group
        
        from openmdao.api import IndepVarComp
        
        group = Group()
        n=1
        k=10
        tar_vector = np.random.rand(3,2)
        comp = IndepVarComp()
        comp.add_output('desptsconstraints', val=np.random.random((k,3)))
        
        group.add_subsystem('IndepVarComp', comp, promotes = ['*'])
        
        
        comp = TiporientationComp(num_nodes=n,k=k,tar_vector=tar_vector)
        group.add_subsystem('desiredpointscomp', comp, promotes = ['*'])
        
        prob = Problem()
        prob.model = group
        
        prob.setup()
        prob.run_model()
        prob.model.list_outputs()

        prob.check_partials(compact_print=True)
        # prob.check_partials(compact_print=False)
        

Adding constraints
~~~~~~~~~~~~~~~~~~
Now, the user is able to import and add the output of the component to be the constraints in optimization.


.. code-block:: python

    '''Constraints'''
        # kinematics, tube geometry constraints
        bccomp = BcComp(num_nodes=num_nodes,k=k)
        diametercomp = DiameterComp()
        tubeclearancecomp = TubeclearanceComp()
        tubestraightcomp = TubestraightComp()
        baseplanarcomp = BaseplanarComp(num_nodes=num_nodes,k=k,equ_paras=equ_paras)
        deployedlenghtcomp = DeployedlengthComp(k=k)
        betacomp = BetaComp(k=k)
        # declare the Openmdao component for the constraints
        tiporientationcomp = TiporientationComp(k=k,tar_vector=tar_vector)

        self.add_subsystem('BetaComp', betacomp, promotes=['*'])
        self.add_subsystem('BcComp', bccomp, promotes=['*'])
        self.add_subsystem('Baseplanarcomp', baseplanarcomp, promotes=['*'])
        self.add_subsystem('DeployedlengthComp', deployedlenghtcomp, promotes=['*'])
        self.add_subsystem('TubestraightComp', tubestraightcomp, promotes=['*'])
        self.add_subsystem('DiameterComp', diametercomp, promotes=['*'])
        self.add_subsystem('TubeclearanceComp', tubeclearancecomp, promotes=['*'])
        # add the new component into the model
        self.add_subsystem('TiporientationComp', tiporientationcomp, promotes=['*'])

        

        self.add_constraint('torsionconstraint', equals=0.)
        self.add_constraint('baseconstraints', lower=0)
        self.add_constraint('deployedlength12constraint', lower=1)
        self.add_constraint('deployedlength23constraint', lower=1)
        self.add_constraint('beta12constraint', upper=-1)
        self.add_constraint('beta23constraint', upper=-1)
        self.add_constraint('diameterconstraint',lower= 0.1)
        self.add_constraint('tubeclearanceconstraint',lower= 0.1,upper=0.16)
        self.add_constraint('tubestraightconstraint',lower= 0)
        # add task-specific constraints
        self.add_constraint('tiporientation', equals=0)



.. toctree::
  :maxdepth: 2
  :titlesonly:
