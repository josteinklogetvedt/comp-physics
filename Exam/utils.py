import numpy as np
import matplotlib.pyplot as plt
import numba as nb

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

newparams = {'lines.linewidth': 2,
             'font.size': 11, 'mathtext.fontset': 'stix',
             'font.family': 'STIXGeneral', 'figure.dpi': 100}
plt.rcParams.update(newparams)

np.random.seed(42)

@nb.njit
def heun_step(y,dt,f,*args):
    y_p = y + dt*f(y,*args)
    return y + (dt/2)*(f(y,*args)+f(y_p,*args))

@nb.njit
def normalize(y): # Nx3  (or N^2 x 3 if the spins are on a lattice)
    res = np.zeros(y.shape)
    for i in range(y.shape[0]):
        res[i] = y[i]/np.sqrt(y[i,0]**2+y[i,1]**2+y[i,2]**2)
    return res

# https://stackoverflow.com/questions/49881468/efficient-way-of-computing-the-cross-products-between-two-sets-of-vectors-numpy
@nb.njit(fastmath=True)    
def calc_cros(vec_1,vec_2): # cross product between two matrices, row-by-row
    res = np.empty(vec_1.shape,dtype=vec_1.dtype)
    
    for i in range(vec_1.shape[0]):
        res[i,0] = vec_1[i,1]*vec_2[i,2] - vec_1[i,2]*vec_2[i,1]
        res[i,1] = vec_1[i,2]*vec_2[i,0] - vec_1[i,0]*vec_2[i,2]
        res[i,2] = vec_1[i,0]*vec_2[i,1] - vec_1[i,1]*vec_2[i,0]
    
    return res

