from sys import path
import numpy as np
import scipy
import os
import time
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None
from ctr_framework.ctrseq_group import CtrseqGroup
from lsdo_viz.api import Problem
from ctr_framework.mesh import trianglemesh
from ctr_framework.initpt import initialize_pt
from ctr_framework.collision_check import collision_check
from ctr_framework.log import log
from ctr_framework.equofplane import equofplane
from ctr_framework.findcircle import findCircle

def seq_opt(num_nodes,viapts_nbr,base,rot,meshfile,pathfile):
    '''
    The function to obtain the a sequence of initial CTR configurations
    that follows the optimized path.

    Parameters
    ----------
    num_nodes : int
        Number of timestep in the numerical integration/ number of links of the robot 
    viapts_nbr : int
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
    k=1
    a = 30
    pt = initialize_pt(viapts_nbr,pathfile)
    pt_pri =  initialize_pt(viapts_nbr * 2,pathfile)
    pt_full =  initialize_pt(100,pathfile)
    p_plane = np.zeros((3,3))
    equ_paras = equofplane(p_plane[0,:],p_plane[1,:],p_plane[2,:])
    center = findCircle(pt[0,1],pt[0,2], \
            pt[-1,1],pt[-1,2],pt[int(viapts_nbr/2),1],pt[int(viapts_nbr/2),2])
    mesh = trianglemesh(num_nodes,k,pt[-1,:],center,meshfile)

    zeta = 0
    rho = 1
    eps_e = 1
    eps_r = 1
    eps_p = 1
    lag = 1
    norm1 = np.linalg.norm(pt[0,:]-pt[-1,:],ord=1.125)
    tol = np.ones((viapts_nbr))*10
    tol[-1] = 5
    t0 = time.time()
    for i in range(0,viapts_nbr,1):
        count = 1
        count1 = 1
        count_error=1
        trigger = 0
        flag = 1
        error = 1
        lag = 1
        while flag==1 or error>tol[i]:
                
            if count >= 10:
                
                if flag==1:
                    multiplier = 20*count1
                    zeta = (1e-3) * multiplier + zeta
                    count1+=1

                prob1 = Problem(model=CtrseqGroup(k=1, num_nodes=num_nodes, a=a, \
                        pt=pt_pri[(i+1)*2-1,:],i=i,target = pt[-1,:], center=center, lag = lag,\
                            zeta=zeta,rho=rho,eps_r=eps_r,eps_p=eps_p, eps_e=eps_e, \
                                pt_full = pt, viapts_nbr=viapts_nbr, meshfile = meshfile,\
                                    rotx_init=rot[0],roty_init=rot[1],rotz_init=rot[2],base = base,count=0,equ_paras = equ_paras,pt_test = pt[-1,:]))
            else:
                
                multiplier = 20 * count
                zeta = 1e-3 * multiplier + zeta
                
                prob1 = Problem(model=CtrseqGroup(k=1, num_nodes=num_nodes, a=a, \
                        pt=pt[i,:],i=i,target = pt[-1,:], center=center, lag = lag,\
                            zeta=zeta,rho=rho,eps_r=eps_r,eps_p=eps_p, eps_e=eps_e,\
                                pt_full = pt, viapts_nbr=viapts_nbr, meshfile = meshfile,\
                                    rotx_init=rot[0],roty_init=rot[1],rotz_init=rot[2],base = base,count=0,equ_paras = equ_paras,pt_test = pt[-1,:]))
            #option 2
            # if flag==1:
            #     multiplier = 10*count1
            #         # zeta = (1e-3) * multiplier1
            #     zeta = (1e-2) * multiplier + zeta
            #     count1+=1
            # prob1 = Problem(model=CtrvizGroup(k=1, num_nodes=num_nodes, a=a, \
            #             pt=pt[i,:],i=i,target = pt[-1,:], center=center, lag = lag,\
            #                 zeta=zeta,rho=rho,eps_r=eps_r,eps_p=eps_p, eps_e=eps_e,\
            #                     pt_full = pt, viapts_nbr=viapts_nbr,\
            #                         rotx_init=rotx,roty_init=roty,base = base,count=0,equ_paras = equ_paras,pt_test = pt[-1,:]))
            prob1.driver = pyOptSparseDriver()
            prob1.driver.options['optimizer'] = 'SNOPT'
            prob1.driver.opt_settings['Verify level'] = 0
            prob1.driver.opt_settings['Major iterations limit'] = 50
            prob1.driver.opt_settings['Minor iterations limit'] = 1000
            prob1.driver.opt_settings['Iterations limit'] = 1000000
            prob1.driver.opt_settings['Major step limit'] = 2.0
            prob1.driver.opt_settings['Major feasibility tolerance'] = 1.0e-4
            prob1.driver.opt_settings['Major optimality tolerance'] = 1.0e-3
            prob1.driver.opt_settings['Minor feasibility tolerance'] = 1.0e-4
            prob1.setup()
            # prob1.run()
            prob1.run_model()
            prob1.run_driver()
            flag,detection = collision_check(prob1['rot_p'],prob1['d2'],prob1['d4'],prob1['d6'],\
                                    prob1['tube_ends'],num_nodes,mesh,k)
            error = prob1['targetnorm']
            log(count,multiplier,i,flag,error)
            if error >= tol[i]:
                lag = lag + rho * prob1['targetnorm']/norm1
                rho = count_error * 5 
                count_error+=1

            elif error<tol[i] and flag==0:
                break
            trigger = 1
            count+=1

        mdict1 = {'points':prob1['integrator_group3.state:p'], 'lota':prob1['lota'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                        'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                        'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                        'initial_condition_dpsi':prob1['initial_condition_dpsi'],'rotx':prob1['rotx'],'roty':prob1['roty'], 'rotz':prob1['rotz'],
                        'loc':prob1['loc'],'rot_p':prob1['rot_p'],'flag':flag, 'detection':detection, 'zeta':zeta, 'dl0':prob1['tube_section_length'] + prob1['beta'],
                        'rho':rho, 'eps_r':eps_r, 'eps_p':eps_p, 'eps_e':eps_e, 
                        'lag':lag
                        }
        scipy.io.savemat('seq_'+str(i)+'.mat',mdict1)
        os.rename('SNOPT_print.out','SNOPT_print'+str(i)+'.out')

    t1 = time.time()
    print(t1-t0)
