import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot 
import scipy.io
from scipy.spatial import KDTree

class trianglemesh:
    # import point cloud
    def __init__(self,filename):
        
        mesh = o3d.io.read_triangle_mesh(filename)
        
        mesh.compute_vertex_normals(normalized=True)
        normals = np.asarray(mesh.vertex_normals)
        ana = np.asarray(mesh.vertices).T
        ana_x = ana[0,:]
        ana_y = ana[1,:]
        ana_z = ana[2,:]
        p = ana[:,:].T
        anatomy = zip(ana_x.ravel(),ana_y.ravel(),ana_z.ravel())
        self.normals = -normals
        self.p = p
        self.ana_x = ana_x
        self.ana_y = ana_y
        self.ana_z = ana_z
        self.tree = KDTree(list(anatomy))

        
    
        
        
if __name__ == '__main__':
    num_nodes = 175
    k=1
    mesh  = trianglemesh()
    tmp = mesh.p
    print(tmp)
   
    
                            
                                  