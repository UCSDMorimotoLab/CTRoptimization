import numpy as np
import scipy
from ctr_framework.design_method.path_opt import path_opt
# from path_opt import path_opt


# Initialize the number of control points and path points
num_cp = 25
num_pt = 100
# User-defined start point and target point
sp = np.array([-10,35,0])
fp = np.array([-10,-33,-103])

# mesh .PLY file
filename = 'trachea.PLY'

path_opt(num_cp,num_pt,sp,fp,filename)



