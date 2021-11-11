import numpy as np




def sim_opt(num_nodes,k,base,rot,meshfile):
    '''
    The function to simultaneously optimize the CTR to follow a number of waypoints and reach the target without collision with anatomy.

    Parameters
    ----------
    num_nodes : int
        Number of timestep in the numerical integration/ number of links of the robot 
    k : int
        Number of via-points, including the target point
    base : Vector or Array (3 by 1)
        The robot base location in 3D space 
    rot : Vector or Array (3 by 1)
        The orientation of the robot base frame about x,y,z axis (rad)
    meshfile : str
        The local path to the mesh file (.ply)
    pathfile : str
        The local path to the path file (.mat)
    '''
    
    

