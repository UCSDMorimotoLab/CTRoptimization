import numpy as np
import scipy
import os


from ctr_framework.design_method.dualsim_opt import dualsim_opt



# Initialize the number of number of links and waypoints
num_nodes = 50
k = 10
num_cases = 2
# robot initial pose 
rotx = np.zeros((num_cases))
roty = np.zeros((num_cases))
rotz = np.zeros((num_cases))
base = np.zeros((num_cases,3))
rotx[0] = 3.2
roty[0] = -0.3
rotz[0] = 0
base[0,:] = np.array([-25,-38,-85]).reshape((3,))
rotx[1] = 3.2
roty[1] = -0.3
rotz[1] = 0
base[1,:] = np.array([-25,-38,-85]).reshape((3,))

# mesh .PLY file
meshfile = 'trachea.PLY'
leftpath = 'leftpath.mat'
rightpath = 'rightpath.mat' 

# run simultaneous optimization
sim_opt(num_nodes,k,num_cases,base,rot,meshfile,leftpath,rightpath)

