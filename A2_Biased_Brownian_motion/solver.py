import numpy as np
import matplotlib.pyplot as plt
import numba as nb   # for faster code

from constants import *

newparams = {'mathtext.fontset': 'stix', 
             'font.family': 'STIXGeneral', 'figure.dpi': 100}  # change font
plt.rcParams.update(newparams)


@nb.njit
def heaviside(t_i): # returns 1 if t_i=3*tau*omega/4
    val = 0
    if t_i >= 0:
        val = 1
    return val

@nb.njit
def U(x,t,tau_omega): # tau_omega = tau*omega
    t_i = t % tau_omega
    f = heaviside(t_i-3*tau_omega/4)
    x_i = x % 1
    return np.where(x_i <alpha,x_i/alpha,(1-x_i)/(1-alpha))*f

@nb.njit
def force(x,t,tau_omega): # = -dU/dx
    t_i = t % tau_omega
    f = heaviside(t_i-3*tau_omega/4)
    x_i = x % 1 
    return np.where(x_i <alpha,(-1/alpha),(1/(1-alpha)))*f  
    
@nb.njit
def Euler_step(x,t,dt,tau_omega,D): # x is an array, t is float
    random = np.random.normal(0,1,len(x))
    return x + dt*force(x,t,tau_omega) + np.sqrt(2*D*dt)*random

@nb.njit
def Brownian_step(x,t,dt,tau_omega,D): # x is an array, t is float
    random = np.random.normal(0,1,len(x))
    return x + np.sqrt(2*D*dt)*random

def time_criterion(dt,dU,r): #not dimensionless time!
    val = 0
    D = thermal_energy/dU
    dt_dimless = dt*dU/(6*np.pi*eta*r*L**2)*electron_charge
    if alpha <= 0.5:
        val = dt_dimless/alpha + 4*np.sqrt(2*D*dt_dimless)
    else:
        val = dt_dimless/(1-alpha) + 4*np.sqrt(2*D*dt_dimless)
    print('Time criterion (<< 1);',val/alpha)
    return val < alpha

@nb.njit 
def solver(t,x_0,tau,dU,r,flashing=True): # x_0 is an array (initial conditions), length=number of particles. t is the time-grid, an array
    x = np.zeros((len(t),len(x_0)))
    potential = np.zeros((len(t),len(x_0)))
    
    D = thermal_energy/dU
    omega = dU/(6*np.pi*eta*r*L**2)*electron_charge  # eV to joule
    tau_omega = tau*omega      # dimensionless period tau
    t_dimless = np.copy(t)*omega
    dt = t_dimless[1]-t_dimless[0]
    
    if not flashing:
        t_dimless = np.ones(len(t))*tau_omega*3/4  # set the heaviside condition to be true for all values of t_dim

    x[0] = x_0/L
    potential[0] = U(x[0],0,tau_omega)
    for i in range(len(t)-1):
        x[i+1] = Euler_step(x[i],t_dimless[i],dt,tau_omega,D)  
        potential[i+1] = U(x[i],t_dimless[i],tau_omega)
    return x*L, potential*dU           # convert back to dimensions


# only extract the last values of positions.
# Potential not gathered. Flashing is always on. 
# step=Euler_step or Brownian_step (without potential)
@nb.njit
def solver_only_last(t,x_0,tau,dU,r,step=Euler_step):                                           
    x = np.copy(x_0)/L
    
    D = thermal_energy/dU
    omega = dU/(6*np.pi*eta*r*L**2)*electron_charge  # transfer from eV to joule
    tau_omega = tau*omega
    t_dimless = np.copy(t)*omega
    dt = t_dimless[1]-t_dimless[0]

    for i in range(len(t)-1):
        x = step(x,t_dimless[i],dt,tau_omega,D)  

    return x*L