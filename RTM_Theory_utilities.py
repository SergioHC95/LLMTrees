#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Theoy of Random Tree Ensemble
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

# with n+1 absorbing states
def transition_matrix_n(N,k=4):

    def off_diag(i,j,k=4):
        if j > i:
            return binom(j-i+k-2,k-2)/binom(j+k-1,k-1)
        else:
            return 0

    T = np.zeros([2*(N+1),2*(N+1)])
    # upper left block
    T[:N+1,:N+1] = np.eye(N+1)
    # lower left block
    T[N+1:,:N+1] = np.zeros([N+1,N+1])
    # lower right block
    for i in range(N+1, 2*(N+1)):
        for j in range(N+1, 2*(N+1)):
            T[i,j] = off_diag(i-(N+1),j-(N+1),k)
    
    T[:,N+1] = np.zeros([2*(N+1)])
    T[:,N+2] = np.zeros([2*(N+1)])

    # upper right block
    for i in range(N+1):
            T[i,N+1+i] = 1/binom(i+k-1,k-1)

    # identify the two fictitious alive 0,1 states to the absorbed 0,1 states
    T[0,N+1] = 1
    T[1,N+2] = 1

    return T

def occupation_transition_n(N,k=4):
    T = transition_matrix_n(N,k)
    B = np.zeros([2*(N+1),2*(N+1)])
    for i in range(2*(N+1)):
        if i < N+3:
            B[i,i] = 0 # 1 if copy the absorbing state down, 0 if not
        else:
            B[i,i] = k
    
    Q = T@B
    return Q


# without absorbing states
def Tij(i,j,k):
    if j >= i:
        return binom(j - i + k - 2, k - 2) / binom(j + k - 1, k - 1)
    else:
        return 0 

def transition_matrix_0(N,k=4):     
    T = np.zeros([N+1,N+1])
    for i in range(N+1):
        for j in range(N+1):
            T[i,j] = Tij(i,j,k)
    return T

# 0,1 absorbing states
def occupation_transition_01(N,k=4):
    T = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            T[i, j] = k*Tij(i, j, k)
    # override the cols 0,1
    T[:, 0] = np.zeros(N+1)
    T[:, 1] = np.zeros(N+1)

    # # copy the absorbing states down
    # T[0, 0] = 1  # absorbing state for 0
    # T[1, 1] = 1  # absorbing state for 1

    return T


def C_n(N,L,k=4):
    N = int(N)
    L = int(L)
    k = int(k)
    
    Q = occupation_transition_n(N,k)
    m1 = np.zeros(2*N+2)
    m1[-1] = 1
    mL = m1
    for l in range(L-1):
        mL = Q @ mL

    e_vec = np.ones(2*N+2)
    e_vec[0] = 0
    e_vec[N+1] = 0
    return e_vec@mL

def C_0(N,L,k=4):
    N = int(N)
    k = int(k)
    
    T = transition_matrix_0(N,k)
    p1 = np.zeros(N+1)
    p1[-1] = 1
    pL = p1
    for l in range(L-1):
        pL = T @ pL
    return k**(L-1) * (1-pL[0])   

def C_01(N,L,k=4):
    N = int(N)
    L = int(L)
    k = int(k)
    
    Q = occupation_transition_01(N,k)
    m1 = np.zeros(N+1)
    m1[-1] = 1
    mL = m1
    for l in range(L-1):
        mL = Q @ mL

    e_vec = np.ones(N+1)
    e_vec[0] = 0
    return e_vec@mL 


def node_dist_n(N,L,k=4):
    N = int(N)
    L = int(L)
    k = int(k)
    
    Q = occupation_transition_n(N,k)
    m1 = np.zeros(2*N+2)
    m1[-1] = 1
    mL = m1
    for l in range(L-1):
        mL = Q @ mL

    e_vec = np.ones(2*N+2)
    e_vec[0] = 0
    e_vec[N+1] = 0
    NL = e_vec@mL

    mL_phys = mL[1:N+1] + mL[N+2:]
    return mL_phys/NL

def node_dist_01(N,L,k=4):
    N = int(N)
    L = int(L)
    k = int(k)
    
    Q = occupation_transition_01(N,k)
    m1 = np.zeros(N+1)
    m1[-1] = 1
    mL = m1
    for l in range(L-1):
        mL = Q @ mL

    e_vec = np.ones(N+1)
    e_vec[0] = 0
    NL = e_vec@mL

    mL_phys = mL[1:]
    return mL_phys/NL

def node_dist_0(N,L,k=4):
    N = int(N)
    L = int(L)
    k = int(k)
    
    T = transition_matrix_0(N,k)
    p1 = np.zeros(N+1)
    p1[-1] = 1
    pL = p1
    for l in range(L-1):
        pL = T @ pL
    pL_phys = pL[1:]/(1-pL[0])
    return pL_phys
