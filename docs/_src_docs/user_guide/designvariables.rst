Design variables
=================

CTR design optimization framework optimizes all the continuous optimization variables,
including the design parameters(tube diameter, tube length, tube curvature), configuration variables(joint values, robot base pose)
and kinematic variables(distal end boundary conditions).


Independent variables
---------------------
The code below shows how the independent variables component
is added in the optimization. Independent variables are set as an model input to the 
optimization. In other words, they can be seen as the initial values for deisgn variables that are 
determined by the user. 

.. code-block:: python

        comp = IndepVarComp(num_nodes=num_nodes,k=k)
        comp.add_output('d1', val=init_guess['d1'])
        comp.add_output('d2', val=init_guess['d2'])
        comp.add_output('d3', val=init_guess['d3'])
        comp.add_output('d4', val=init_guess['d4'])
        comp.add_output('d5', val=init_guess['d5'])
        comp.add_output('d6', val=init_guess['d6'])
        comp.add_output('kappa', shape=(1,3), val=init_guess['kappa'])
        comp.add_output('tube_section_length',shape=(1,3),val=init_guess['tube_section_length'])
        comp.add_output('tube_section_straight',shape=(1,3),val=init_guess['tube_section_straight'])
        comp.add_output('lota', shape=(k,3),val=init_guess['lota'])
        comp.add_output('beta', shape=(k,3),val=init_guess['beta']+0.01)
        comp.add_output('initial_condition_dpsi', shape=(k,3), val=init_guess['initial_condition_dpsi'])
        comp.add_output('rotx',val=init_guess['rotx'])
        comp.add_output('roty',val=init_guess['roty'])
        comp.add_output('rotz',val=init_guess['rotz'])
        comp.add_output('loc',shape=(3,1),val=init_guess['loc'])
        self.add_subsystem('input_comp', comp, promotes=['*'])
        
      

Adding design variables
-----------------------
After the independent variables are set, the user are able to define and add
the design varialbes to the optimization. The user can also set the upper and lower 
bound to those design variables as their choice.


.. code-block:: python

        "Deisgn variables"
        # tube diameter
        self.add_design_var('d1',lower= 0.2 , upper=3.5)       # ID_1
        self.add_design_var('d2',lower= 0.2, upper=3.5)        # OD_1
        self.add_design_var('d3',lower= 0.2, upper=3.5)        # ID_2
        self.add_design_var('d4',lower= 0.2, upper=3.5)        # OD_2
        self.add_design_var('d5',lower= 0.2, upper=3.5)        # ID_3
        self.add_design_var('d6',lower= 0.2, upper=3.5)        # OD_3
        # tube length
        self.add_design_var('tube_section_length',lower=0)      # total tube length
        self.add_design_var('tube_section_straight',lower=0)    # tube length of straight section
        # configuration variables
        self.add_design_var('lota')                             # tip rotation
        self.add_design_var('beta', upper=-1)                   # tube translation
        # tube curvature  
        self.add_design_var('kappa', lower=0)
        # Robot base frame
        self.add_design_var('rotx')     # rotation about x-axis
        self.add_design_var('roty')     # rotation about y-axis
        self.add_design_var('rotz')     # rotation about z-axis
        self.add_design_var('loc')      # 3-D position of the base



.. toctree::
  :maxdepth: 2
  :titlesonly:
