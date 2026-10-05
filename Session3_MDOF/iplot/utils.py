import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as linalg

def frequency_response_function(M:np.array, C:np.array, K:np.array, f= np.linspace(0,2,100), i=1,j=1):    
    if not isinstance(f, np.ndarray):
        f = np.array([f])
    s = f*np.pi*2*1j
    
    M_s2 = np.outer(M,(s**2)).reshape(M.shape[0], -1, len(s))
    C_s = np.outer(C,(s**1)).reshape(M.shape[0], -1, len(s))
    K_s = np.outer(K,(s**0)).reshape(M.shape[0], -1, len(s))

    H_ij = []
    for s_i in range(len(s)):
        H_ij.append(np.linalg.inv((M_s2+C_s+K_s)[:,:,s_i])[i,j])
    
    return H_ij

def calc_eigenvalues_eigenvectors(M:np.array,K:np.array):    
    lamb, phi = linalg.eig(K,M)

    return lamb, phi
def build_3DOF_matrices(m:float,c:float,c1:float,k:float,k1:float):     
    M_mat=np.array([[m, 0, 0],
           [0, m, 0],
           [0, 0, m]])
    
    ###BEGIN SOLUTION
    
    K_mat=np.array([[k1+k, -k, 0],
           [-k, 2*k, -k],
           [0, -k, k]])
    C_mat=np.array([[c1+c, -c, 0],
            [-c, 2*c, -c],
            [0,-c,c]])
    ###END SOLUTION
    return M_mat, C_mat, K_mat
