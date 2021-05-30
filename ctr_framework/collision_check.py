import numpy as np
from mesh import trianglemesh

def collision_check(rot_p,d2,d4,d6,tube_ends,num_nodes,mesh,k):
    rot_p = rot_p.reshape(num_nodes,k,3)
    query,normals = mesh.nn(rot_p)
    dis = rot_p-query
    
    # cross section radii
    tube1 = np.zeros((num_nodes,k))
    tube2 = np.zeros((num_nodes,k))
    tube3 = np.zeros((num_nodes,k))
    cross_section = np.zeros((num_nodes,k))
    tube1 = (tube_ends[:,:,0] - tube_ends[:,:,1])
    tube2 = (tube_ends[:,:,1] - tube_ends[:,:,2])
    tube3 = tube_ends[:,:,2]
    
    # self.cross_section = cross_section
    cross_section = tube1  * d2/2  + tube2 * d4/2 + tube3 * d6/2
    # check robot is inside or not
    inner_product = np.einsum("ijk,ijk->ij", dis, normals)
    detection = inner_product * tube_ends[:,:,0]
    if np.any(detection < 0):
        flag = 1
    else:
        flag = 0 
    # idx_negative = np.where((inner_product<=0))
    # idx_negative = np.array(idx_negative)

    return flag, detection

# if __name__ == '__main__':
    

