import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot 
import scipy.io
from scipy.spatial import KDTree

class trianglemesh:
    # import point cloud
    def __init__(self,num_nodes,k,meshfile):
        mesh = o3d.io.read_triangle_mesh(meshfile)
        self.num_nodes = num_nodes
        self.k = k
        self.mesh = mesh
        mesh.compute_vertex_normals(normalized=True)
        normals = np.asarray(mesh.vertex_normals)
        ana = np.asarray(mesh.vertices).T
        ana_x = ana[0,:]
        ana_y = ana[1,:]
        ana_z = ana[2,:]
        p = ana[:,:].T 
        anatomy = zip(ana_x.ravel(),ana_y.ravel(),ana_z.ravel())
        
        self.normals = normals
        self.p = p
        self.ana_x = ana_x
        self.ana_y = ana_y
        self.ana_z = ana_z
        self.tree = KDTree(list(anatomy))

    def nn(self,query):
        #pcd = self.pcd
        # mesh = self.mesh
        kk = self.k
        tree = self.tree
        ana_x = self.ana_x
        ana_y = self.ana_y
        ana_z = self.ana_z
        normals = self.normals
        num_nodes = self.num_nodes
        # query = query['points']
        # ctr_x = query[:,:,0]
        # ctr_y = query[:,:,1]
        # ctr_z = query[:,:,2]
        
        nearest = np.array(tree.query(query,k=1))
        # normals = np.zeros((num_nodes,kk,3))
        # normals = np.asarray(mesh.vertex_normals)[40:,:]
        # print(normals.shape)
        # print(nearest.shape)
        nearest_normals = np.zeros((num_nodes,kk,3))
        nearest_normals = np.reshape(normals[nearest[1,:,:].astype(int).flatten(),:],(num_nodes,kk,3))  
        # nearest_normals[:,:,1] = np.reshape(normals[nearest[1,:,:].astype(int).flatten(),1],(num_nodes,kk))
        # nearest_normals[:,:,2] = np.reshape(normals[nearest[1,:,:].astype(int).flatten(),2],(num_nodes,kk))
        # mesh.compute_triangle_normals()
        # print(ana_x.shape)
        # print(normals.shape)
        # normal_rot = np.dot(Rotx,norm/al_vectors.T)
        # print(normal_rot[:,1000:2000]) 
        # o3d.visualization.draw_geometries([pcd])
        # o3d.visualization.draw_geometries([mesh])
        nearest_pts = np.zeros((num_nodes,kk,3))
        # nearest_pts[:,:,0] = np.reshape(ana_x[nearest[1,:,:].astype(int).flatten()],(num_nodes,1))  
        # nearest_pts[:,:,1] = np.reshape(ana_y[nearest[1,:,:].astype(int).flatten()],(num_nodes,1))
        # nearest_pts[:,:,2] = np.reshape(ana_z[nearest[1,:,:].astype(int).flatten()],(num_nodes,1))
        nearest_pts[:,:,0] = np.reshape(ana_x[nearest[1,:,:].astype(int).flatten()],(num_nodes,kk))  
        nearest_pts[:,:,1] = np.reshape(ana_y[nearest[1,:,:].astype(int).flatten()],(num_nodes,kk))
        nearest_pts[:,:,2] = np.reshape(ana_z[nearest[1,:,:].astype(int).flatten()],(num_nodes,kk))
        
        return nearest_pts, -nearest_normals
        # print('triangle',np.asarray(mesh.triangle_normals))
        # print('vertex',np.asarray(mesh.vertex_normals))
    
    
if __name__ == '__main__':
    num_nodes = 175
    k=1
    mesh  = trianglemesh(num_nodes,k)
    normal = mesh.normals
    vec=mesh.tar_vector()
    print(vec)
  
    
                            
                                  