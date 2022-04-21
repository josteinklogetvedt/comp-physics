import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy import sparse
from scipy.sparse.linalg import eigs

from plotting import *

#import psutil #to write out memory capacity
#import time

@nb.njit
def COO_matrix(U_ind,classified_grid,row,column,data):
    n = U_ind.shape[0]
    r = np.zeros(5*n,dtype=np.int32)
    c = np.zeros(5*n,dtype=np.int32)
    d = np.zeros(5*n,dtype=np.float32)
    r[:n] = row
    c[:n] = column
    d[:n] = data
    m = n
    for i in range(n):
        k = U_ind[i,0]
        l = U_ind[i,1]
        if classified_grid[k,l-1] == 2:
            r[m] = i
            c[m] = i-1
            d[m] = -1.0
            m += 1
        if classified_grid[k,l+1] == 2:
            r[m] = i
            c[m] = i+1
            d[m] = -1.0
            m += 1
        if classified_grid[k-1,l] == 2:
            j = np.where((U_ind[:,0]==k-1) & (U_ind[:,1]==l))[0][0]
            r[m] = i
            c[m] = j
            d[m] = -1.0
            m += 1
        if classified_grid[k+1,l] == 2:
            j = np.where((U_ind[:,0]==k+1) & (U_ind[:,1]==l))[0][0]
            r[m] = i
            c[m] = j
            d[m] = -1.0
            m += 1
    return r[:m],c[:m],d[:m]

def make_matrix(classified_grid,delta):
    U_ind = np.argwhere(classified_grid==2).astype(np.uint32)
    n = U_ind.shape[0]
    C = sparse.eye(n,dtype=np.float32,format='coo')*4
    r,c,d = COO_matrix(U_ind,classified_grid,C.row,C.col,C.data)
    d = d/delta**2
    B = sparse.coo_matrix((d,(r,c)),shape=(n,n),dtype=np.float32).tocsr()
    return B, U_ind
    
def find_eigenval_and_vec(B,val):
    w, v = eigs(B,k=val,which='SR')
    return w,v      # w is eigenvalues, and v are the eigenmodes

# ------ plot contour --------
def plot_mode(l,l_max,mode=0,c_file=False,func=make_matrix,val=10,savename=False):
    assert(l <= l_max)
    xv,yv = initialize_grid(l_max)
    classified_grid = 0
    if c_file:
        classified_grid = np.load(c_file)['arr_0']
    else:
        frac = Koch_fractal(l)
        classified_grid = classify_grid(frac,xv,yv)
    delta = abs(xv[0,0]-xv[0,1])
    B, U_ind = func(classified_grid,delta)

    w,v = find_eigenval_and_vec(B,val)
    
    U = np.zeros((classified_grid.shape[0],classified_grid.shape[0]),dtype=np.float64)
    U[U_ind[:,0],U_ind[:,1]] = np.real(v[:,mode])
    plot_contour(U,xv,yv,l,savename)

#plot_mode(3,4,0,c_file='Data/c_grid34.npz')

# ------ plot d_N --------
def plot_d_N_23and33():
    cg_33 = np.load('Data/c_grid33.npz')['arr_0']
    xv,yv = initialize_grid(3)
    delta = abs(xv[0,0]-xv[0,1])
    B,U_ind = make_matrix(cg_33,delta)
    w_33,v = find_eigenval_and_vec(B,700)  # the first 700 eigenvalues

    cg_23 = np.load('Data/c_grid23.npz')['arr_0']
    B,U_ind = make_matrix(cg_23,delta)
    w_23,v = find_eigenval_and_vec(B,500)

    w_23 = np.sort(np.sqrt(np.real(w_23)))
    d_N_23 = w_23**2/(4*np.pi) - np.arange(0,len(w_23))
    plt.plot(w_23,d_N_23,label=r'$(2,3)$',color='green')
    w_33 = np.sort(np.sqrt(np.real(w_33)))
    d_N_33 = w_33**2/(4*np.pi) - np.arange(0,len(w_33))
    plt.plot(w_33,d_N_33,label=r'$(3,3)$',color='red')
    plt.grid()
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Delta N(\omega)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    #plt.savefig('d_N_23and33.pdf')
    plt.show()

#plot_d_N()
#plot_d_N_23and33()


# ----- determine d for l=inf -------
from scipy.optimize import curve_fit

def plot_d_curve_fit(l,d):
    func = lambda x,a,b: a/(b+np.exp(-x))   #trial function, logistic growth
    popt, pcov = curve_fit(func,l,d)

    print('Approximated value for d(inf);',func(np.inf,popt[0],popt[1]))
    
    x = np.linspace(l[0],12,50)
    plt.scatter(l,d,label='Data',marker='x',c='b')
    plt.plot(x,func(x,popt[0],popt[1]),label='Curve fit',color='black',linewidth=2)
    plt.grid()
    plt.legend()
    plt.xlabel(r'$l$')
    plt.ylabel(r'$d$')
    #plt.savefig('curve_fit_d.pdf')
    plt.show()

d = np.asarray([1.413,1.512,1.55,1.553])
l = np.array([2,3,4,5])
#plot_d_curve_fit(l,d) 
