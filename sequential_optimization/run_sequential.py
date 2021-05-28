import numpy as np
import scipy
import os
from openmdao.api import pyOptSparseDriver
from openmdao.api import ScipyOptimizeDriver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pyOptSparseDriver = None
from ctrseq_group import CtrseqGroup
from lsdo_viz.api import Problem
from mesh import trianglemesh
from initpt import initialize_pt
from collision_check import collision_check
from log import log
import shutil
import time
from equofplane import equofplane
from findcircle import findCircle


viapts_nbr=10
num_nodes = 50
k = 1
a = 30
pt = initialize_pt(viapts_nbr)
pt_pri =  initialize_pt(viapts_nbr * 2)
pt_full =  initialize_pt(100)


d1 = 0.65
d2 = 0.88
d3 = 1.076
d4 = 1.296
d5 = 1.470
d6 = 2.180
kappa_init = np.array([0.0061, 0.0131,0.0021]).reshape((1,3))
tube_length_init = np.array([200, 120,65]).reshape((1,3)) + 100
tube_straight_init = np.array([150, 80,35]).reshape((1,3)) + 50
alpha_init = np.zeros((k,3))
'heartcase01'
alpha_init[:,0] = -np.pi/2
alpha_init[:,1] = np.pi/1.5
alpha_init[:,2] = -np.pi/3
beta_init = np.zeros((k,3))
beta_init[:,0] = -280
beta_init[:,1] = -205
beta_init[:,2] = -155

init_dpsi = np.random.random((k,3)) *0.01
dl0 = tube_length_init + beta_init 

rotx_ = 1e-10 
roty_ = 1e-10
rotz_ = 1e-10
loc = np.ones((3,1)) * 1e-5


mdict = {'alpha':alpha_init, 'beta':beta_init,'kappa':kappa_init,
        'tube_section_straight':tube_straight_init,'tube_section_length':tube_length_init,
        'd1':d1, 'd2':d2, 'd3':d3, 'd4':d4, 'd5':d5, 'd6':d6, 'initial_condition_dpsi':init_dpsi,
        'rotx':rotx_,'roty':roty_ ,'rotz':rotz_ , 'loc':loc,
        }
scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/initial.mat',mdict)


'final trachea model'
rotx = 3.14
roty = 0
rotz = 0
base = np.array([-10,35,20]).reshape((3,1))
p_plane = np.array([[-10,35,20],[-12,20,20],\
                    [-20,15,20]])



equ_paras = equofplane(p_plane[0,:],p_plane[1,:],p_plane[2,:])



pt = initialize_pt(viapts_nbr)
pt_pri =  initialize_pt(viapts_nbr * 2)
center = findCircle(pt[0,1],pt[0,2], \
        pt[-1,1],pt[-1,2],pt[int(viapts_nbr/2),1],pt[int(viapts_nbr/2),2])
mesh = trianglemesh(num_nodes,k,pt[-1,:],center)

zeta = 0
rho = 1
eps_e = 1
eps_r = 1
eps_p = 1
lag = 1
norm1 = np.linalg.norm(pt[0,:]-pt[-1,:],ord=1.125)
tol = np.ones((viapts_nbr))*10
tol[-1] = 10
'step 2: sequential optimization'
t0 = time.time()
for i in range(0,viapts_nbr,1):
    
    #####################
    # flag = 0
    #####################
    count = 1
    count1 = 1
    count_error=1
    trigger = 0
    flag = 1
    error = 4
    
    lag = 1
    while flag==1 or error>tol[i]:
            
        if count >= 10:
            
            if flag==1:
                multiplier = 20*count1
                zeta = (1e-3) * multiplier + zeta
                count1+=1

            'dimensionless'
            prob1 = Problem(model=CtrseqGroup(k=1, num_nodes=num_nodes, a=a, \
                    pt=pt_pri[(i+1)*2-1,:],i=i,target = pt[-1,:], center=center, lag = lag,\
                        zeta=zeta,rho=rho,eps_r=eps_r,eps_p=eps_p, eps_e=eps_e, \
                            pt_full = pt, viapts_nbr=viapts_nbr,dl0=dl0,\
                                rotx_init=rotx,roty_init=roty,rotz_init=rotz,base = base,count=0,equ_paras = equ_paras,pt_test = pt[-1,:]))
        else:
            
            multiplier = 20 * count
            zeta = 1e-3 * multiplier + zeta
            
            prob1 = Problem(model=CtrseqGroup(k=1, num_nodes=num_nodes, a=a, \
                    pt=pt[i,:],i=i,target = pt[-1,:], center=center, lag = lag,\
                        zeta=zeta,rho=rho,eps_r=eps_r,eps_p=eps_p, eps_e=eps_e,\
                            pt_full = pt, viapts_nbr=viapts_nbr, dl0=dl0,\
                                rotx_init=rotx,roty_init=roty,rotz_init=rotz,base = base,count=0,equ_paras = equ_paras,pt_test = pt[-1,:]))
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

        mdict_flag = {'points':prob1['integrator_group3.state:p'], 'alpha':prob1['alpha'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                    'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                    'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                    'initial_condition_dpsi':prob1['initial_condition_dpsi'],'rotx':prob1['rotx'],'roty':prob1['roty'], 'rotz':prob1['rotz'],
                    'loc':prob1['loc'],'rot_p':prob1['rot_p'],'flag':flag
                    }                                                       
        scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/sequential_data/flag_htest18_'+str(i)+'_'+str(count)+'.mat',mdict_flag)
        trigger = 1
        count+=1

    mdict1 = {'points':prob1['integrator_group3.state:p'], 'alpha':prob1['alpha'], 'beta':prob1['beta'],'kappa':prob1['kappa'],
                    'tube_section_straight':prob1['tube_section_straight'],'tube_section_length':prob1['tube_section_length'],
                    'd1':prob1['d1'], 'd2':prob1['d2'], 'd3':prob1['d3'], 'd4':prob1['d4'], 'd5':prob1['d5'], 'd6':prob1['d6'],
                    'initial_condition_dpsi':prob1['initial_condition_dpsi'],'rotx':prob1['rotx'],'roty':prob1['roty'], 'rotz':prob1['rotz'],
                    'loc':prob1['loc'],'rot_p':prob1['rot_p'],'flag':flag, 'detection':detection, 'zeta':zeta,
                    'rho':rho, 'eps_r':eps_r, 'eps_p':eps_p, 'eps_e':eps_e, 
                    'lag':lag
                    }
    scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/results/sequential_data/seq_htest18_'+str(i)+'.mat',mdict1)
    
    os.rename('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/SNOPT_print.out','/home/fred/Desktop/ctr_optimization/code_opts_seqv2/SNOPT_print'+str(i)+'.out')

t1 = time.time()
print(t1-t0)
