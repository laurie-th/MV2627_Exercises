import numpy as np
from scipy import linalg

def build_3DOF_matrices(m:float,c:float,c1:float,k:float,k1:float): 
    """
    Implement a function that computes the build the system matrix for a 3DOF matrix

    Arguments:
    m,k,k1,c,c1 -- parameters of the MDOF system

    Returns:
    M_mat : The M matrix from the system equation
    K_mat : The K matrix from the system equation
    C_mat : The C matrix from the system equation
  
    Tips use np.array([[],[],[]]) to define your 2D matrix
    check documentation FMI 'Numpy.array()'
    """  
    # here is an example of how to set a matrix 
    
    M_mat=np.array([[m, 0, 0],
           [0, m, 0],
           [0, 0, m]])
    K_mat=np.array([[k1+k, -k, 0],
           [-k, 2*k, -k],
           [0, -k, k]])
    C_mat=np.array([[c1+c, -c, 0],
            [-c, 2*c, -c],
            [0,-c,c]])
    return M_mat, C_mat, K_mat

def frequency_response_function(M:np.array, C:np.array, K:np.array, f= np.linspace(0,2,100), i=1,j=1):
    """
    A function that computes the Frequency Response Function from the system matrices

    Arguments:
    M,C,K -- system matrices of the MDOF system
    f -- the frequency for which the repsponse needs to be calculated
    i,j -- the specific element of the FRF matrix to be computed.

    Return: H_ij -- the i-th row and j-th column element of the FRF function at f
    """
    
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
    """
    A function that computes the eigenvalues and eigenvectors
    based on the system parameter by reworking the equations of motion to a generalized eigenvalue problem

    Arguments:
    M,K -- system matrices of the MDOF system

    Returns:
    lamb -- The eigenvalues of the conservative eigenvalue problem
    psi -- The eigenvectors of the conservative eigenvalue problem
  
    
    """   
    
    lamb, psi = linalg.eig(K,M)
    # sort the eigenvalues and eigenvectors from smallest to largest
    idx = np.abs(lamb).argsort()[::1]
    lamb = lamb[idx]
    psi = psi[:,idx]

    return lamb, psi

def lambda_to_resonance_frequency(lamb:np.array):
    """
    A function that computes the resonance frequency from the eigenvalues

    Arguments:
    lamb -- eigenvalues of the conservative eigenvalue problem

    Returns:
    w_d -- The resonance frequency of the system
  
    
    """   
    w_d=np.sqrt(lamb)/2/np.pi
    return w_d