from utils import *
from scipy.signal import argrelextrema


# Code for solutions to task a) and b)

kappa = 0.1    # = d_z/J
beta = 0.1     # = mu_s*B_0/J
# use dimensionless time t_dimless = gamma*B*t

# including only d_z > 0, means B = 0. Have also T = 0 here
@nb.njit
def normalize_1D(y):
    return y/np.linalg.norm(y)  

def analytical(S_0,omega,t): # analytical solution to task a)
    S_x = S_0[0]*np.cos(omega*t) - S_0[1]*np.sin(omega*t)
    S_y = S_0[1]*np.cos(omega*t) + S_0[0]*np.sin(omega*t)
    return S_x, S_y

@nb.njit
def single_dynamics(S,alpha):  # d_t S = f() function
    F = np.asarray([0,0,2*(kappa/beta)*S[2]])
    S_cross_F = np.cross(S,F)
    return (-1)*(1/(1+alpha**2))* (S_cross_F + alpha* np.cross(S,S_cross_F))

def time_solver(S_0,t,alpha):
    S = np.zeros((len(t),len(S_0)),dtype=S_0.dtype)
    S[0] = normalize_1D(S_0)
    dt = t[1]-t[0]
    
    for i in range(1,len(t)):
        S[i] = normalize_1D(heun_step(S[i-1],dt,single_dynamics,alpha))

    return S
