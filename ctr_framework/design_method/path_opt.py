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

from ctr_framework.bspline_group import BsplineGroup
from ctr_framework.bspline_3d_comp import BsplineComp


def path_opt(num_cp,num_pt,sp,fp,filename):

    r2 = 0.1
    r1 = 1

    prob = Problem(model=BsplineGroup(num_cp=num_cp,num_pt=num_pt,
                                        sp=sp,fp=fp,
                                            r2=r2,r1=r1,filename=filename))

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major iterations limit'] = 400 
    prob.driver.opt_settings['Minor iterations limit'] = 1000
    prob.driver.opt_settings['Iterations limit'] = 1000000
    prob.driver.opt_settings['Major step limit'] = 2.0
    prob.setup()
    prob.run_model()
    prob.run_driver()


