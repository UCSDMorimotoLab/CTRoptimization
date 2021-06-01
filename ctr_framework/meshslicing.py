import numpy as np

def meshslicing(via_pt,mesh,center):
    
    
    # tree = KDTree(mesh[:,1:])
    # find 10 closet point with respect to each via-points
    # dd, ii = tree.query(via_pt[1:], k=1)
    # slope = (mesh[ii,2] - via_pt[2]) / (mesh[ii,1] - via_pt[1]) 
    
    slope = (center[1] - via_pt[2]) / (center[0] - via_pt[1])
    # compute the line and the sign
    # d = (x-x1)*(y2-y1)-(y-y1)*(x2-x1)
    index = []
    
    for i in range(mesh.shape[0]):
        # d = (mesh[i,1]-via_pt[1])*(mesh[ii,2]-via_pt[2])-(mesh[i,2]-via_pt[2])*(mesh[ii,1]-via_pt[1])
        d = (mesh[i,1]-via_pt[1])*(center[1]-via_pt[2])-(mesh[i,2]-via_pt[2])*(center[0]-via_pt[1])
        if slope<=0 and d>=0:
            index.append(i)
        elif slope > 0 and d<=0:
            index.append(i)

    return np.asarray(index)
