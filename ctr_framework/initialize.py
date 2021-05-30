import numpy as np

def initialize_bspline(sp,fp,num_cp,num_pt):

    # control points
    t_c = np.linspace(0, 1, num_cp, endpoint=True)
    x_c = sp[0] + t_c * (fp[0]-sp[0])
    y_c = sp[1] + t_c * (fp[1]-sp[1])
    z_c = sp[2] + t_c * (fp[2]-sp[2])
    cp = np.array([x_c,y_c,z_c]).T
    # path points
    t_p = np.linspace(0, 1, num_pt, endpoint=True)
    x_p = sp[0] + t_p * (fp[0]-sp[0])
    y_p = sp[1] + t_p * (fp[1]-sp[1])
    z_p = sp[2] + t_p * (fp[2]-sp[2])
    pt = np.array([x_p,y_p,z_p]).T

    return cp,pt