import numpy as np
# import scipy
# import os
# from openmdao.api import pyOptSparseDriver
# from openmdao.api import ScipyOptimizeDriver
# try:
#     from openmdao.api import pyOptSparseDriver
# except:
#     pyOptSparseDriver = None

# from ctrviz_group import CtrvizGroups
from ctr_framework.ctrsimul_group import CtrsimulGroup
# from lsdo_viz.api import Problem
from ctr_framework.mesh_simul import trianglemesh
from ctr_framework.initpt import initialize_pt
from ctr_framework.collision_check import collision_check
import time
from ctr_framework.equofplane import equofplane
from ctr_framework.fibonacci_sphere import fibonacci_sphere
from ctr_framework.log import log



def sim_opt(num_nodes,k,base,rot,meshfile):
    '''
    The function to simultaneously optimize the CTR to follow a number of waypoints and reach the target without collision with anatomy.

    Parameters
    ----------
    num_nodes : int
        Number of timestep in the numerical integration/ number of links of the robot 
    k : int
        Number of via-points, including the target point
    base : Vector or Array (3 by 1)
        The robot base location in 3D space 
    rot : Vector or Array (3 by 1)
        The orientation of the robot base frame about x,y,z axis (rad)
    meshfile : str
        The local path to the mesh file (.ply)
    pathfile : str
        The local path to the path file (.mat)
    '''
    
    

