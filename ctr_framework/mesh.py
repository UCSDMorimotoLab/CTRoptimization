import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial import KDTree
from meshslicing import meshslicing

class trianglemesh:
    # import point cloud
    def __init__(self,num_nodes,k,via_pt,center):
        # mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/final/heart_case03_final1.ply")
        # mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/Heart/final/case04_sfinal.ply")
        mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/trachea_1191v_nt.PLY")
        # mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/final/heart7.ply")
        # self.mesh = o3d.io.read_triangle_mesh("/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/torsionally_compliant/ctr_optimization/mesh/lumen4.PLY")
        self.num_nodes = num_nodes
        self.k = k
        self.mesh = mesh
        
        mesh.compute_vertex_normals(normalized=True)
        # vertex_normals = np.dot(Rotz,np.dot(Rotx,np.asarray(mesh.vertex_normals).T)).T
        normals = np.asarray(mesh.vertex_normals)
        ana = np.asarray(mesh.vertices).T
                # print(ana.shape)
        # ana = np.dot(Rotx,np.asarray(mesh.vertices).T)
        # ana_x = ana[0,:]
        # ana_y = ana[1,:]
        # ana_z = ana[2,:]
        ana_x = ana[0,:]
        ana_y = ana[1,:]
        ana_z = ana[2,:]
        p = ana[:,:].T 

        anatomy = zip(ana_x.ravel(),ana_y.ravel(),ana_z.ravel())
        

        'Meshslicing'
        index = meshslicing(via_pt,p,center)
        # sliced_p = p[index,:]
        
        self.index = index
        self.normals = normals[index,:]
        self.normals_nn = normals
        self.p = p[index,:]
        self.ana_x = ana_x
        self.ana_y = ana_y
        self.ana_z = ana_z
        self.tree = KDTree(list(anatomy))

        # import ctr_backbone point
        # self.query = scipy.io.loadmat('/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/torsionally_compliant/ctr_optimization/code/results/ctr_mesh.mat')
    def nn(self,query):
        #pcd = self.pcd
        # mesh = self.mesh
        kk = self.k
        tree = self.tree
        ana_x = self.ana_x
        ana_y = self.ana_y
        ana_z = self.ana_z
        normals = self.normals_nn
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
    def tar_vector(self):
        ana_x = self.ana_x
        ana_y = self.ana_y
        ana_z = self.ana_z
        tree = self.tree
        query = np.zeros((1,3))
        query[:,0] = ana_x[40]
        query[:,1] = ana_y[40]
        query[:,2] = ana_z[40]

        nearest = np.array(tree.query(query,k=2))
        nearest_pts = np.zeros((2,3))
        nearest_pts[:,0] = ana_x[nearest[1,:].astype(int).flatten()]  
        nearest_pts[:,1] = ana_y[nearest[1,:].astype(int).flatten()]
        nearest_pts[:,2] = ana_z[nearest[1,:].astype(int).flatten()]
        ###### Only for k = 1(1 configuration) right now        
        return nearest_pts.T

    def visualization(self):
        index = self.index
        ana_x = self.ana_x
        ana_y = self.ana_y
        ana_z = self.ana_z
        mesh = self.mesh
       
        return ana_x, ana_y, ana_z
        
    # o3d.visualization.draw_geometries([mesh])
if __name__ == '__main__':
    num_nodes = 175
    k=1
    # pt = np.array([-10.2280,34.0452,-5.9356])
    # pt = np.array([-10.3953,28.8986,-41.5950])
    pt = scipy.io.loadmat('/home/fred/Desktop/ctr_optimization/code_opts_baseang/trajectory_optimization/trajectory_optimization/pt.mat')
    pt = np.asarray(pt['pt'])
    # print(pt)
    # mesh  = trianglemesh(num_nodes,k,pt)
    # p = mesh.p
    # index = mesh.index
    # new_p = p[index,:]
    center = np.array([-70,10])
    for i in range(0,pt.shape[0],9):
        mesh  = trianglemesh(num_nodes,k,pt[i,:],center)
        new_p = mesh.p
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(new_p[:,0], new_p[:,1], new_p[:,2])
        ax.scatter3D(pt[i,0], pt[i,1], pt[i,2])
        
        plt.show()
    # normal = mesh.normals
    # vec=mesh.tar_vector()
    
    # minpts,idx = mesh.min_nn(np.random.random((num_nodes,k,3)))
    # _,normals = mesh.nn(np.random.random((num_nodes,k,3)))
    
    # mesh.visualization()
    # print(np.asarray(normals).shape)
    
                            
                                  