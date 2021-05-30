import math
import numpy as np
import matplotlib.pyplot as plt

def fibonacci_sphere(samples,r,layer):
    sphere = np.zeros((layer,samples,3))
    for j in range(layer):
        step = r / layer + j
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y) * step # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append((x, y, z))
            # print(points)
        sphere[j,:,:] = np.asarray(points)    
    sphere = sphere.reshape(-1,3)
    return sphere 

if __name__ == '__main__':
    layer = 10
    r = 10
    samples = 100
    # points = np.zeros((layer,samples,3))
    points = fibonacci_sphere(samples,r,layer)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # print(points.shape)
    # ax.plot3D(points[:,0], points[:,1], points[:,2], 'gray')
    # print(points)
    ax.scatter3D(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap='Greens')
    sphere = points.reshape(-1,3)
    idx = np.random.randint(sphere.shape[0], size = 100)
    # ax.scatter3D(sphere[idx,0], sphere[idx,1], sphere[idx,2], c=sphere[idx,2], cmap='hot_r')

    plt.show()