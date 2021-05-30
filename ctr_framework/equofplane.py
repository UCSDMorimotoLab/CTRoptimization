import numpy as np

def equofplane(p1,p2,p3):
    # two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # vector normal
    cp  = np.cross(v1,v2)
    q, r, s = cp

    #
    t = np.dot(cp, p3)

    return np.array([q,r,s,t])