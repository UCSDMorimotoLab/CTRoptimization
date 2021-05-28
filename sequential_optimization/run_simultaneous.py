import numpy as np
import scipy
import os
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None

# from ctrviz_group import CtrvizGroups
from ctrsimul_group import CtrsimulGroup
from lsdo_viz.api import Problem
from mesh_simul import trianglemesh
from initpt import initialize_pt
from collision_check import collision_check
import time
from equofplane import equofplane
from fibonacci_sphere import fibonacci_sphere
from log import log

via_pts=10
num_nodes = 50
k = 10
mesh = trianglemesh(num_nodes,k)
a = 30


# robot initial pose trachea
rotx = 3.14
roty = 0
rotz = 0
base = np.array([-10,35,20]).reshape((3,1))

base = np.array([-10,55,20]).reshape((3,1))

base = np.array([-10,43,20]).reshape((3,1))
pt = initialize_pt(via_pts)
pt_pri =  initialize_pt(via_pts * 2)
# find 3 points on the plane
# p_plane = np.array([[-13.2501,-22.5262,110.735],[-12.6813,-26.3715,98.0471],\
#                     [-19.8698,-25.6478,103.586]])
p_plane = np.array([[-10,35,20],[-12,20,20],\
                    [-20,15,20]])
equ_paras = equofplane(p_plane[0,:],p_plane[1,:],p_plane[2,:]) 
norm1 = np.linalg.norm(pt[0,:]-pt[-1,:],ord=1.125)


# pt_pri =  initialize_pt(via_pts * 2)
# opt_tol = [1e-2,1e-3]
'step 3: final optimization'
k = 10
k_ = 1
alpha_ = np.zeros((k,3))
beta_ = np.zeros((k,3))
initial_condition_dpsi_ = np.zeros((k,3))
lag_ = np.zeros((k,1))
rho_ = np.zeros((k,1))
zeta_ = np.zeros((k,1))

count = 0
t0 = time.time()
for i in range(via_pts):
    # configs = scipy.io.loadmat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/sequential_data/seq_htest18_'+str(i)+'.mat')
    # configs = scipy.io.loadmat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/trachea_/seq_htest18_'+str(i)+'.mat')
    # configs = scipy.io.loadmat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/heart_final_case03_01/seq_htest18_'+str(i)+'.mat')
    configs = scipy.io.loadmat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/heart_final_case04_02/seq_htest18_'+str(i)+'.mat')
    # configs = scipy.io.loadmat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/heart_final_experiments_02/seq_htest18_'+str(i)+'.mat')
    alpha_[count,:] = configs['alpha']
    beta_[count,:] = configs['beta']
    initial_condition_dpsi_[count,:] = configs['initial_condition_dpsi']
    lag_[count,:] = configs['lag']
    rho_[count,:] = configs['rho']
    zeta_[count,:] = configs['zeta']
    count = count+1
mdict1 = {'alpha':alpha_, 'beta':beta_,'kappa':configs['kappa'], 'rho':rho_, 'lag':lag_, 'zeta':zeta_,
                    'tube_section_straight':configs['tube_section_straight'],'tube_section_length':configs['tube_section_length'],
                    'd1':configs['d1'], 'd2':configs['d2'], 'd3':configs['d3'], 'd4':configs['d4'], 'd5':configs['d5'], 'd6':configs['d6'],
                    'initial_condition_dpsi':initial_condition_dpsi_, 'rotx':configs['rotx'],'roty':configs['roty'],'rotz':rotz,  ########### <- check
                    'eps_r':configs['eps_r'], 'eps_p':configs['eps_p'], 'eps_e':configs['eps_e'], 'loc':configs['loc'],
                    }
scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_simul/results/simul.mat',mdict1)

flag=1
error=np.ones((k,1))
i=0
lag = np.ones((k,1))
multiplier_zeta = 1 
multiplier_rho = 1
while flag==1 or error[-1]>5:

    if flag==1 and i>=0:
        multiplier_zeta = 50*i
        zeta_ = multiplier_zeta + zeta_
    
    prob1 = Problem(model=CtrsimulGroup(k=k, k_=k_, num_nodes=num_nodes, a=a, i=1, \
                        pt=pt[:,:],target = pt[-1,:],\
                        pt_full = pt, viapts_nbr=via_pts, zeta = zeta_, rho=rho_, lag=lag_,\
                        rotx_init=rotx,roty_init=roty, rotz_init=rotz,base = base,equ_paras = equ_paras))
    i+=1
    prob1.driver = pyOptSparseDriver()
    prob1.driver.options['optimizer'] = 'SNOPT'
    # prob1.driver.opt_settings['Verify level'] = 0
    prob1.driver.opt_settings['Major iterations limit'] = 30 #1000
    prob1.driver.opt_settings['Minor iterations limit'] = 1000
    prob1.driver.opt_settings['Iterations limit'] = 1000000
    prob1.driver.opt_settings['Major step limit'] = 2.0
    prob1.driver.opt_settings['Major feasibility tolerance'] = 1.0e-4
    prob1.driver.opt_settings['Major optimality tolerance'] = 1.0e-3
    prob1.driver.opt_settings['Minor feasibility tolerance'] = 1.0e-4

    prob1.setup()
    prob1.run_model()
    prob1.run_driver()
    flag,detection = collision_check(prob1['rot_p'],prob1['d2'],prob1['d4'],prob1['d6'],\
                                prob1['tube_ends'],num_nodes,mesh,k)
    error = prob1['targetnorm']

    if error[-1,:] >= 5:
        multiplier_rho = error
        rho_ = multiplier_rho + rho_
        lag_ = lag_ + (rho_) * error/norm1 
        # lag = lag + rho * error/norm1 
    mdict2 = {'points':prob1['integrator_group3.state:p'], 'alpha':prob1['alpha'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                    'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                    'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                    'initial_condition_dpsi':prob1['initial_condition_dpsi'], 'loc':prob1['loc'], 'rotx':prob1['rotx'], 'roty':prob1['roty'],
                    'rotz':prob1['rotz'],
                    'rho':rho_,'lag':lag_,'zeta':zeta_, 'eps_r':configs['eps_r'], 'eps_p':configs['eps_p'], 'eps_e':configs['eps_e'],
                    'error':prob1['targetnorm'], 'tip_position':prob1['desptsconstraints'],
                    }
    # scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2_simul/results/simul_heartcase01'+str(i)+'.mat',mdict2)

    scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/heart_final_case04_02/simul_heartcase04_'+str(i+2)+'.mat',mdict2)
    # scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/heart_final_case04_01/simul_heartcase04_test'+str(i)+'.mat',mdict2)
    # scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/heart_final_experiments_02/simul_heartcase01_f'+str(i)+'.mat',mdict2)
    
t1 = time.time()
print(t1-t0)
