CTR Kinematics
==============


Three main steps:
~~~~~~~~~~~~~~~~~


1. Boundary value problem
-------------------------
The differential equations can be written as:

.. math:: \ddot{\psi}_{i} = \dfrac{k_{ib}}{k_{it}k_{b}}\sum\limits_{j=1}^{n}k_{jb}\kappa_{i}\kappa_{j}\sin(\psi_{i}-\psi_{j})
with the boundary conditions:

.. math:: \psi_{i}(0) = \alpha_{i} - \beta_{i}\dot{\psi}_{i}(0),
.. math:: \dot{\psi}_{i}(L_{i}+\beta_{i}) = 0.

where the :math:`\psi`, :math:`\dot{\psi}` are the tube angle and tube torsion, :math:`k_{ib}` is the bending stiffness of tube :math:`i`,
:math:`k_{it}` is the torsional stiffness of tube :math:`i`, :math:`\kappa_{i}` is the curvature if tube i and :math:`\alpha_{i}` 
and :math:`\beta_{i}` are the joint variables --- translation and rotation of tube :math:`i`.

Alternative formulation as an initial value problem that is used in the current framework is presented belwo. With 
this problem formulation, the optimization problem will has a unique solution since the tip rotation and tube base rotation
will now have a unique mapping and this is not gurantee with the BVP problem when the CTR snapping issue occurs.
The boundary conditions are turned into two initial conditions at distal end and can be written as:

.. math:: \psi_{i}(L_{i}+\beta_{i}) = \phi_{i},
.. math:: \dot{\psi}_{i}(L_{i}+\beta_{i}) = 0.

2. Robot curvature vector
-------------------------
The deformed curvature vector of the robot along the robot backbone can be found by:


.. math:: u = K^{-1}\sum\limits_{i=1}^{n}K_{i}(R_{\psi_i}u_{i}^{*}-\dot{\psi_{i}}e_{3}),
where :math:`K_{i}` is a :math:`3\times3` stiffness tensor of tube :math:`i`, :math:`K 
= \sum_{i=1}^{n}K_{i}`, and :math:`u_{i}^{*}` is the pre-curvature vector of tube :math:`i`.

3. Robot backbone position in 3D-space
--------------------------------------
Two differential equations for reconstructing the backbone position are as follows:

.. math:: \mathbf{\dot{R}} = \mathbf{R}\mathbf{\hat{u}},\\
.. math:: \mathbf{\dot{p}} = \mathbf{R}\mathbf{e}_{3},

Reference
----------
D. C. Rucker, R. J. Webster III, G. S. Chirikjian, and N. J. Cowan,“Equilibrium conformations of concentric-tube continuum robots,
”TheInternational journal of robotics research, vol. 29, no. 10, pp. 1263–1280, 2010

Bergeles, Christos, and Pierre E. Dupont. "Planning stable paths for concentric tube robots.
"2013 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2013.


