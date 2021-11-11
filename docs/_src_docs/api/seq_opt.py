from sys import path
import numpy as np
# import scipy
# import os
# import time
# from openmdao.api import pyOptSparseDriver
# from openmdao.api import ScipyOptimizeDriver
# try:
#     from openmdao.api import pyOptSparseDriver
# except:
#     pyOptSparseDriver = None
from ctr_framework.ctrseq_group import CtrseqGroup
# from lsdo_viz.api import Problem
from ctr_framework.mesh import trianglemesh
from ctr_framework.initpt import initialize_pt
from ctr_framework.collision_check import collision_check
from ctr_framework.log import log
from ctr_framework.equofplane import equofplane
from ctr_framework.findcircle import findCircle

def seq_opt(num_nodes,viapts_nbr,base,rot,meshfile,pathfile):
    '''
    The function to obtain the a sequence of initial CTR configurations
    that follows the optimized path.

    Parameters
    ----------
    num_nodes : int
        Number of timestep in the numerical integration/ number of links of the robot 
    viapts_nbr : int
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
    