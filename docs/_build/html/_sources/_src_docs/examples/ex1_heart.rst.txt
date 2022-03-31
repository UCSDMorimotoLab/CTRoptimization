Clinial example 1: Heart biopsy
===============================

Optimization problem
--------------------
In this example, the user can follow the instructions below to run the full optimization process through: path, sequential, and simultaneous 
optimization step.

The run files for each step are as follows.
The user-defined parameter are the anatomical model (.ply format), starting point, target point 
and the number of control points, path points.
A B-spline curve will be optimized by the function below.

.. code-block:: python

    import numpy as np
    import scipy
    from ctr_framework.design_method.path_opt import path_opt


    # Initialize the number of control points and path points
    num_cp = 25
    num_pt = 100
    # User-defined start point and target point
    sp = np.array([-23,-8,-85])
    fp = np.array([87,-27,-193])

    # mesh .PLY file
    filename = 'heart.ply'

    path_opt(num_cp,num_pt,sp,fp,filename)

Once the collision-free path is found, the sequental optimization needs to be performed in order to 
get a better initial guesses for the simultaneous CTR optimization problem. The codes are as follow

.. code-block:: python

    import numpy as np
    import scipy
    
    from ctr_framework.design_method.seq_opt import seq_opt

    #########################################
    ############## initialization ###########
    #########################################

    # number of waypoints
    viapts_nbr=10
    # number of links                              
    num_nodes = 50
    # Extract the waypoints from optimized path
    pt = initialize_pt(viapts_nbr)
    pt_pri =  initialize_pt(viapts_nbr * 2)
    pt_full =  initialize_pt(100)


    # initial robot configuration
    # Tube 1(inner tube) ID, OD
    d1 = 0.65
    d2 = 0.88
    # Tube 2 
    d3 = 1.076
    d4 = 1.296
    # Tube 3(outer tube)
    d5 = 1.470
    d6 = 2.180
    # Tube curvature (kappa)
    kappa_init = np.array([0.0061, 0.0131,0.0021]).reshape((1,3))
    # The length of tubes
    tube_length_init = np.array([200, 120,65]).reshape((1,3)) + 100
    # The length of straight section of tubes
    tube_straight_init = np.array([150, 80,35]).reshape((1,3)) + 50
    # joint variables
    alpha_init = np.zeros((k,3))
    alpha_init[:,0] = -np.pi/2
    alpha_init[:,1] = np.pi/1.5
    alpha_init[:,2] = -np.pi/3
    beta_init = np.zeros((k,3))
    beta_init[:,0] = -280
    beta_init[:,1] = -205
    beta_init[:,2] = -155
    # initial torsion 
    init_dpsi = np.random.random((k,3)) *0.01
    rotx_ = 1e-10 
    roty_ = 1e-10
    rotz_ = 1e-10
    loc = np.ones((3,1)) * 1e-5

    mdict = {'alpha':alpha_init, 'beta':beta_init,'kappa':kappa_init,
            'tube_section_straight':tube_straight_init,'tube_section_length':tube_length_init,
            'd1':d1, 'd2':d2, 'd3':d3, 'd4':d4, 'd5':d5, 'd6':d6, 'initial_condition_dpsi':init_dpsi,
            'rotx':rotx_,'roty':roty_ ,'rotz':rotz_ , 'loc':loc,
            }
    scipy.io.savemat('initial.mat',mdict)

    # Base frame

    base = np.array([-23,-8,-57]).reshape((3,1))
    rot = np.array([3.14,0,0]).reshape((3,1))
    
    # mesh .PLY file
    meshfile = 'heart.ply'
    pathfile = 'path.mat'

    seq_opt(num_nodes,viapts_nbr,base,rot,meshfile,pathfile)


The second step serves as an initial guesses for the final step, which is the patient-speific 
simultaneous optimization. In this step, the optimizer optimizes k robot configurations simultaneously
to obtain a robot design and safe motion plan.

.. code-block:: python

    import numpy as np
    
    from ctr_framework.design_method.sim_opt import sim_opt



    # Initialize the number of number of links and waypoints
    num_nodes = 50
    k = 10
    # robot initial pose 
    base = np.array([-23,-8,-57]).reshape((3,1))
    rot = np.array([3.14,0,0]).reshape((3,1))

    # mesh .PLY file
    meshfile = 'heart.ply'
    pathfile = 'path.mat'

    # run simultaneous optimization
    sim_opt(num_nodes,k,base,rot,meshfile,pathfile)


 




