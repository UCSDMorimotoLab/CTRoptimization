import numpy as np
import scipy.io

def initialize_pt(k,pathfile):
    
    path = scipy.io.loadmat(pathfile)
    
    temp = np.asarray(path['pt'])
    pt = np.zeros((k,3))
    nbr = int(temp.shape[0]/k)
    pt = temp[int(nbr-1)::nbr]
    mdict = {'pt':pt}
    # scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/trajectory_optimization/trajectory_optimization/post_pt.mat',mdict)
    return pt
    

if __name__ == '__main__':
    import scipy.io
    k = 100
    pt=initialize_pt(k)
    print(pt)
    idx = np.array([0,4,8,15,24,38,80,90,95,99])
    idx = np.array([0,0,2,4,6,8,12,15,20,24,30,38,50,80,85,90,93,95,97,99])
    pt = pt[idx,:]
    
    mdict = {'pt':pt}
    scipy.io.savemat('/home/fred/Desktop/ctr_optimization/code_opts_seqv2/trajectory_optimization/trajectory_optimization/post_pt.mat',mdict)
    
