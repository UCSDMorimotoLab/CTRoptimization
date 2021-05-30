import open3d as o3d
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot 
import scipy.io
from scipy.spatial import KDTree

class trianglemesh:
    # import point cloud
    def __init__(self,num_nodes,k):
        # mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/trachea_1191v_nt.PLY")
        # mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/final/heart_case03_final1.ply")
        # mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/Heart/final/case04_sfinal.ply")
        mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/final/heart7.ply")
        # self.mesh = o3d.io.read_triangle_mesh("/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/torsionally_compliant/ctr_optimization/mesh/lumen4.PLY")
        # mesh = o3d.io.read_triangle_mesh("/home/fred/Desktop/ctr_optimization/mesh/trachea_1191v_sz1p5.ply")
        self.num_nodes = num_nodes
        self.k = k
        self.mesh = mesh
        '''theta = 3.6 
        Rotx = [[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]]
        thetaz = 1.5708 * 2
        Rotz = [[np.cos(thetaz),-np.sin(thetaz),0],[np.sin(thetaz),np.cos(thetaz),0],[0,0,1]]
        mesh.compute_vertex_normals(normalized=True)
        # vertex_normals = np.dot(Rotz,np.dot(Rotx,np.asarray(mesh.vertex_normals).T)).T
        vertex_normals = np.asarray(mesh.vertex_normals).T
        
        vertex_normals[:,0] = vertex_normals[:,0] + 15
        vertex_normals[:,1] = vertex_normals[:,1] + 25.5
        vertex_normals[:,2] = vertex_normals[:,2] - 110.7245

        normals = np.dot(Rotx,np.dot(Rotz,vertex_normals[:,:])).T

        # ana[0,40:] = ana[0,:] + 15
        # ana[1,40:] = ana[1,:] + 28.5
        # ana[2,40:] = ana[2,:] - 110.7245
        # ana = np.dot(Rotz,np.dot(Rotx,np.asarray(mesh.vertices).T))
        ana = np.asarray(mesh.vertices).T
        ana[0,:] = ana[0,:] + 15
        ana[1,:] = ana[1,:] + 28.5
        ana[2,:] = ana[2,:] - 110.7245
        ana = np.dot(Rotx,np.dot(Rotz,ana))
        # print(ana.shape)
        # ana = np.dot(Rotx,np.asarray(mesh.vertices).T)'''
        
        mesh.compute_vertex_normals(normalized=True)
        # vertex_normals = np.dot(Rotz,np.dot(Rotx,np.asarray(mesh.vertex_normals).T)).T
        normals = np.asarray(mesh.vertex_normals)
        ana = np.asarray(mesh.vertices).T
                # print(ana.shape)
        # ana = np.dot(Rotx,np.asarray(mesh.vertices).T)
        ana_x = ana[0,:]
        ana_y = ana[1,:]
        ana_z = ana[2,:]
        p = ana[:,:].T 

        'J shape'
        # theta = 1.5708
        '''theta = 1.5708
        Rotx = [[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]]
        mesh.compute_vertex_normals(normalized=True)
        vertex_normals = np.asarray(mesh.vertex_normals).T
        # vertex_normals[:,0] = vertex_normals[:,0] + 15
        # vertex_normals[:,1] = vertex_normals[:,1] + 25.5
        # vertex_normals[:,2] = vertex_normals[:,2] - 110.7245
        normals = np.dot(Rotx,vertex_normals).T


        ana = np.asarray(mesh.vertices).T
        # ana[0,:] = ana[0,:] + 15
        # ana[1,:] = ana[1,:] + 28.5
        # ana[2,:] = ana[2,:] - 110.7245
        ana = np.dot(Rotx,ana)
        # print(ana.shape)
        # ana = np.dot(Rotx,np.asarray(mesh.vertices).T)

        ana_x = ana[0,40::5]
        ana_y = ana[1,40::5]
        ana_z = ana[2,40::5]
        p = ana[:,:].T'''
        anatomy = zip(ana_x.ravel(),ana_y.ravel(),ana_z.ravel())
        
        self.normals = normals
        self.p = p
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
        ana_x = self.ana_x
        ana_y = self.ana_y
        ana_z = self.ana_z
        mesh = self.mesh
        # pcd = self.pcd
        # pcd estimation 
        
        # fig = pyplot.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(ana_x[::10],ana_y[::10],ana_z[::10])
        # ax.scatter(-5.5,-8.9,100)
        # mesh.compute_vertex_normals(normalized=True)
        # mesh.compute_triangle_normals()
        # mesh_normals = np.asarray(mesh.triangle_normals)
        # vertex_normals = np.asarray(mesh.vertex_normals)
        # mesh_vertices = np.asarray(mesh.vertices)
        #print(mesh_vertices.shape)
        return ana_x, ana_y, ana_z
        
        # print('triangle',mesh_normals)
        # print('vertex',np.linalg.norm(vertex_normals[0,:]))
        # Save joint values and tube parameters to .mat file
        # mdict = {'vertex_normals':vertex_normals,'mesh_vertices':mesh_vertices}
        # scipy.io.savemat('D:/Desktop/Fred/CTR/CTR optimization/CTR/Inverse_kinematics/jointvalues/jointvalue_distance_001.mat',mdict)
        # scipy.io.savemat('/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/torsionally_compliant/ctr_optimization/code/results/vertex_normal.mat',mdict)

        # print('pcdnormals',normal_vectors)
        # print('vertex',np.asarray(mesh.vertex_normals))
        # o3d.visualization.draw_geometries([mesh])
        # plt.zticks(np.arange(-100, 100, step=20))
        # print(ana_x[::10].shape)
        # print(vertex_normals.shape)
        # ax.quiver(mesh_vertices[50:-1:10,0],mesh_vertices[50:-1:10,0],mesh_vertices[50:-1:10,0],vertex_normals[50:-1:10,0],vertex_normals[50:-1:10,1],vertex_normals[50:-1:10,2],length=3,normalize=True)
        # ax.scatter(ana_x,ana_y,ana_z)
        # ax.scatter(ana_x[nearest[1,:,:,:].astype(int).flatten()],ana_y[nearest[1,:,:,:].astype(int).flatten()],ana_z[nearest[1,:,:,:].astype(int).flatten()])
        # ax.scatter(ana_x[nearest[1,:,:].astype(int).flatten()],ana_y[nearest[1,:,:].astype(int).flatten()],ana_z[nearest[1,:,:].astype(int).flatten()],marker='H',c='purple')
        # ax.scatter(ana_x[min_idx],ana_y[min_idx],ana_z[min_idx],c='r',marker='*',s=100)
        # pyplot.show()
    # o3d.visualization.draw_geometries([mesh])
if __name__ == '__main__':
    num_nodes = 175
    k=1
    mesh  = trianglemesh(num_nodes,k)
    normal = mesh.normals
    vec=mesh.tar_vector()
    print(vec)
    # minpts,idx = mesh.min_nn(np.random.random((num_nodes,k,3)))
    # _,normals = mesh.nn(np.random.random((num_nodes,k,3)))
    
    # mesh.visualization()
    # print(np.asarray(normals).shape)
    
                            
                                  