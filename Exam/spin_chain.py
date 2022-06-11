from utils import *

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
from matplotlib.colors import Normalize
from cycler import cycler


# Code for solutions to task c), d), e) and f)
# different cases with and without damping
# and with and without PBC (periodic boundary condition)

kappa = 0.3   # = d_z/J
beta = 0.4    # = mu_s*B_0/J

@nb.njit
def sum_NN_chain_no_PBC(S):  # sum over nearest neighbour, NO periodic boundary conditon
    res = np.zeros((S.shape[0]-2,S.shape[1]))
    for i in range(1,S.shape[0]-1):
        res[i-1] = S[i-1] + S[i+1]
    return res

@nb.njit
def sum_NN_chain_w_PBC(S):  # sum over nearest neighbour, WITH periodic boundary conditon
    res = np.zeros((S.shape[0],S.shape[1]))
    res[0] = S[-1] + S[1]
    res[-1] = S[-2] + S[0]
    for i in range(1,S.shape[0]-1):
        res[i] = S[i-1] + S[i+1]
    return res

@nb.njit
def spin_chain_no_PBC(S,alpha,J):  # derivative function d_t S = f(S), include sum over N.N and d_z >0, NO PBC.

    F = np.zeros(S.shape)
    F[0] = S[1]
    F[-1] = S[-2]
    F[1:-1,:] = sum_NN_chain_no_PBC(S)
    F = np.sign(J)*(2/beta)*F
    F[:,2] += 2*(kappa/beta)*S[:,2]
    S_cross_F = calc_cros(S,F)
    q = (-1)*(1/(1+alpha**2))* (S_cross_F + alpha* calc_cros(S,S_cross_F))

    return q

@nb.njit
def spin_chain_w_PBC(S,alpha,J):  # d_t S = f(S), include sum over N.N and d_z > 0, WITH PBC.

    F = sum_NN_chain_w_PBC(S)
    F = np.sign(J)*(2/beta)*F
    F[:,2] += 2*(kappa/beta)*S[:,2]
    S_cross_F = calc_cros(S,F)
    q = (-1)*(1/(1+alpha**2))* (S_cross_F + alpha* calc_cros(S,S_cross_F))

    return q


def time_solver(S_0,t,f,alpha,J):  # general input of f()
    S = np.zeros((len(t),*S_0.shape),dtype=S_0.dtype)
    S[0] = normalize(S_0)
    dt = t[1]-t[0]
    
    for i in range(1,len(t)):
        S[i] = normalize(heun_step(S[i-1],dt,f,alpha,J))

    return S

