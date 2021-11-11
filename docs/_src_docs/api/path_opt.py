import numpy as np
import scipy
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
from openmdao.api import Problem, pyOptSparseDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None

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

    r2 = 10
    r1 = 1

    prob = Problem(model=BsplineGroup(num_cp=num_cp,num_pt=num_pt,
                                        sp=sp,fp=fp,
                                            r2=r2,r1=r1,filename=meshfile))

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major iterations limit'] = 400 
    prob.driver.opt_settings['Minor iterations limit'] = 1000
    prob.driver.opt_settings['Iterations limit'] = 1000000
    prob.driver.opt_settings['Major step limit'] = 2.0
    prob.setup()
    prob.run_model()
    prob.run_driver()

    # save  
    mdict = {'pt':prob['pt'],'cp':prob['cp']}
    scipy.io.savemat('path.mat',mdict)



