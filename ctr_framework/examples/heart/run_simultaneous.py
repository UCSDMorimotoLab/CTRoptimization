import numpy as np
import scipy
import os


from ctr_framework.design_method.sim_opt import sim_opt



# Initialize the number of number of links and waypoints
num_nodes = 50
k = 10
# robot initial pose 
base = np.array([-23,-8,-57]).reshape((3,1)) # 3D base postion
rot = np.array([3.14,0,0]).reshape((3,1))   # base roation about x,y,z axis
# mesh .PLY file
meshfile = 'heart.ply'
pathfile = 'path.mat'

# run simultaneous optimization
sim_opt(num_nodes,k,base,rot,meshfile,pathfile)

