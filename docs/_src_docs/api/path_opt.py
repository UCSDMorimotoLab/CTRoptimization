import numpy as np
# import scipy
# from openmdao.api import pyOptSparseDriver
# from openmdao.api import ScipyOptimizeDriver
# from openmdao.api import Problem, pyOptSparseDriver
# try:
#     from openmdao.api import pyOptSparseDriver
# except:
#     pyOptSparseDriver = None

from ctr_framework.bspline_group import BsplineGroup
from ctr_framework.bspline_3d_comp import BsplineComp


def path_opt(num_cp,num_pt,sp,fp,meshfile):
    '''
    The function to use 3D-Bspline function to optimize a collision-free path for the robot to follow.



    Parameters
    ----------
    num_cp : int
        Number of B-spline path control points
    num_pt : int
        Number of Bspline path points
    sp : Vector or Array (1 by 3)
        The start point of the Bspline in 3D space
    fp : Vector or Array (1 by 3)
        The fibal target point of the Bspline in 3D space
    meshfile : str
        The local path to the mesh file (.ply)
    '''

    



