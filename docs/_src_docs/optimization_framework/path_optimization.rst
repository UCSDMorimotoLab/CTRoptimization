Step 1: Path optimization
=========================

Optimization problem
--------------------
In this tutorial, we will show you how to construct the path optimization problem in OpenMDAO and how to use the function.

The Bspline group is as follows:

.. code-block:: python

    import numpy as np
    import openmdao.api as om 
    from openmdao.api import Problem, Group, ExecComp, IndepVarComp
    from ozone.api import ODEIntegrator
    from startpoint_comp import StartpointComp
    from finalpoint_comp import FinalpointComp
    from mesh_path import trianglemesh
    from initialize import initialize_bspline
    from bspline_3d_comp import BsplineComp, get_bspline_mtx
    from pt_comp import PtComp
    from signedpt_comp import SignedptComp
    from ptequdistant1_comp import Ptequdistant1Comp
    from ptequdistant2_comp import Ptequdistant2Comp
    from pathobjective_comp import PathobjectiveComp


    class BsplineGroup(om.Group):
        def initialize(self):
            self.options.declare('filename')
            self.options.declare('r2')
            self.options.declare('r1')
            self.options.declare('sp')
            self.options.declare('fp')
            self.options.declare('num_cp', default=25, types=int)
            self.options.declare('num_pt', default=100, types=int)
            
            

        def setup(self):
            filename = self.options['filename']
            r2 = self.options['r2']
            r1 = self.options['r1']
            sp = self.options['sp']
            fp = self.options['fp']
            num_cp = self.options['num_cp']
            num_pt = self.options['num_pt']
            
            # mesh processing
            mesh  = trianglemesh(filename)
            p_ = mesh.p
            normals = mesh.normals
            
            comp = IndepVarComp(num_cp=num_cp,num_pt=num_pt)
            
            c_points,p_points = initialize_bspline(sp,fp,num_cp,num_pt)
            

            comp.add_output('cp', val=c_points)
            
            self.add_subsystem('input_comp', comp, promotes=['*'])
            jac = get_bspline_mtx(num_cp,num_pt)
            bspline_comp =  BsplineComp(
            num_cp=num_cp,
            num_pt=num_pt,
            jac=jac,
            in_name='cp',
            out_name='pt',
            )

            self.add_subsystem('Bspline_comp', bspline_comp, promotes=['*'])
            startpointcomp = StartpointComp(num_cp=num_cp)
            Finalpointcomp = FinalpointComp(num_cp=num_cp)
            self.add_subsystem('Startpointcomp', startpointcomp, promotes=['*'])
            self.add_subsystem('Finalpointcomp', Finalpointcomp, promotes=['*'])
            pt_comp = PtComp(num_pt=num_pt,p_=p_,normals=normals)
            self.add_subsystem('Pt_comp',pt_comp,promotes=['*'])
            signedpt_comp = SignedptComp(num_pt=num_pt,p_=p_,normals=normals)
            self.add_subsystem('Signedpt_comp',signedpt_comp,promotes=['*'])
            ptequdistant1_comp = Ptequdistant1Comp(num_pt=num_pt)
            self.add_subsystem('ptequdistant1_comp', ptequdistant1_comp, promotes=['*'])
            ptequdistant2_comp = Ptequdistant2Comp(pt_=p_points,num_pt=num_pt)
            self.add_subsystem('ptequdistant2_comp',ptequdistant2_comp,promotes=['*'])
            norm1 = np.linalg.norm(sp-fp,ord=1.125)
            pathobjective_comp = PathobjectiveComp(r2=r2,r1=r1/norm1)
            self.add_subsystem('pathobjective_comp',pathobjective_comp,promotes=['*'])

            
            # Design variable
            self.add_design_var('cp')

            # Constraints
            self.add_constraint('startpoint_constraint',equals=sp)
            self.add_constraint('finalpoint_constraint',equals=fp)
            
            
            # Objectives
            self.add_objective('path_objective')

After the group is built, we now can solve the path optimization problem by running the optimizer and code below:

.. code-block:: python

    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from openmdao.api import pyOptSparseDriver
    from openmdao.api import ScipyOptimizeDriver
    from openmdao.api import Problem, pyOptSparseDriver
    try:
        from openmdao.api import pyOptSparseDriver
    except:
        pyOptSparseDriver = None

    from bspline_group import BsplineGroup
    from bspline_3d_comp import BsplineComp

    # Initialize the number of control points and path points
    num_cp = 25
    num_pt = 100

    'heart04'
    # Define the start point and final point (target)
    sp = np.array([-23,-8,-85])
    fp = np.array([87,-27,-193])
    r2 = 0.1
    r1 = 1
    # mesh .PLY file
    filename = '/mesh.ply'

    prob = Problem(model=BsplineGroup(num_cp=num_cp,num_pt=num_pt,
                                        sp=sp,fp=fp,
                                            r2=r2,r1=r1,
                                                filename=filename))
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    # prob.driver.opt_settings['Verify level'] = 0
    prob.driver.opt_settings['Major iterations limit'] = 400 
    prob.driver.opt_settings['Minor iterations limit'] = 1000
    prob.driver.opt_settings['Iterations limit'] = 1000000
    prob.driver.opt_settings['Major step limit'] = 2.0
    prob.setup()
    prob.run_model()
    prob.run_driver()


    print('Path points')
    print(prob['pt'])
    print('Control points')
    print(prob['cp'])


.. toctree::
  :maxdepth: 2
  :titlesonly:
