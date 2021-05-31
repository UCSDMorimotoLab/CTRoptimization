import numpy as np
import scipy
import os
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None

# from ctrviz_group import CtrvizGroups
from ctrsimul_group import CtrsimulGroup
from lsdo_viz.api import Problem
from mesh_simul import trianglemesh
from initpt import initialize_pt
from collision_check import collision_check
import time
from equofplane import equofplane
from fibonacci_sphere import fibonacci_sphere
from log import log
from sim_opt import sim_opt



# Initialize the number of number of links and waypoints
num_nodes = 50
k = 10
# robot initial pose 
base = np.array([-10,35,20]).reshape((3,1))
rot = np.array([3.14,0,0]).reshape((3,1))

# mesh .PLY file
meshfile = '/home/fred/Desktop/ctr_optimization/mesh/Heart/final/case04_sfinal.ply'

# run simultaneous optimization
sim_opt(num_nodes,k,base,rot,meshfile)

