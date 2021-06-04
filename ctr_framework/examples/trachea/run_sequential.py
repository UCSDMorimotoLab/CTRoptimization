import numpy as np
import scipy
import os
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None
from ctr_framework.ctrseq_group import CtrseqGroup
from lsdo_viz.api import Problem
from ctr_framework.mesh import trianglemesh
from ctr_framework.initpt import initialize_pt
from ctr_framework.collision_check import collision_check
from ctr_framework.log import log
import shutil
import time
from ctr_framework.equofplane import equofplane
from ctr_framework.findcircle import findCircle
from ctr_framework.design_method.seq_opt import seq_opt

#########################################
############## initialization ###########
#########################################

# number of waypoints
viapts_nbr=10
k = 1
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

base = np.array([-10,35,20]).reshape((3,1))
rot = np.array([3.14,0,0]).reshape((3,1)) 
p_plane = np.array([[-10,35,20],[-12,20,20],\
                    [-20,15,20]])
# mesh .PLY file
meshfile = 'trachea.ply'

seq_opt(num_nodes,viapts_nbr,base,rot,meshfile)


