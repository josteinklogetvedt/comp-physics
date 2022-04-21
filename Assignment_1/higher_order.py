import numpy as np
import numba as nb
from scipy import sparse

from plotting import *
from helmholts_eq import plot_mode


@nb.njit
def get_neigbours(classified_grid,k,l):
    neighbours = np.zeros(4,dtype=np.uint32)
    neighbours[0] = classified_grid[k,l-1]
    neighbours[1] = classified_grid[k,l+1]
    neighbours[2] = classified_grid[k-1,l]
    neighbours[3] = classified_grid[k+1,l]
    return neighbours

@nb.njit
def COO_matrix_HO(U_ind,classified_grid):
    space = 14
    n = U_ind.shape[0]
    r = np.zeros(space*n,dtype=np.int32)
    c = np.zeros(space*n,dtype=np.int32)
    d = np.zeros(space*n,dtype=np.float32)
    m = 0
    for i in range(n):
        k = U_ind[i,0]
        l = U_ind[i,1]
        neighbours = get_neigbours(classified_grid,k,l)
        if neighbours[0]==1:   #assuming a fine enough grid so that there are atleast three gridpoints between boundaries.
            r[m]=i; c[m]=i; d[m]= -5/3
            r[m+1]=i; c[m+1]=i+1; d[m+1]= 1/2
            r[m+2]=i; c[m+2]=i+2; d[m+2]= 1/3
            m += 3
            if classified_grid[k,l+3]==2:
               r[m]=i; c[m]=i+3; d[m]=-1/12 
               m += 1
        elif neighbours[1]==1:
            r[m]=i; c[m]=i; d[m]= -5/3
            r[m+1]=i; c[m+1]=i-1; d[m+1]= 1/2
            r[m+2]=i; c[m+2]=i-2; d[m+2]= 1/3
            m += 3
            if classified_grid[k,l-3]==2:
                r[m]=i; c[m]=i-3; d[m]=-1/12 
                m += 1
        else:
            r[m]=i; c[m]=i; d[m]= -5/2
            r[m+1]=i; c[m+1]=i+1; d[m+1]= 4/3
            r[m+2]=i; c[m+2]=i-1; d[m+2]= 4/3
            m += 3
            if classified_grid[k,l-2]==2:
                r[m]=i; c[m]=i-2; d[m]=-1/12;
                m+=1
            if classified_grid[k,l+2]==2:
                r[m]=i; c[m]=i+2; d[m]=-1/12;
                m+=1
                
        if neighbours[2]==1:
            r[m]=i; c[m]=i; d[m]= -5/3
            j_p = np.where((U_ind[:,0]==k+1) & (U_ind[:,1]==l))[0][0]
            r[m+1]=i; c[m+1]=j_p; d[m+1]= 1/2
            j_pp = np.where((U_ind[:,0]==k+2) & (U_ind[:,1]==l))[0][0]
            r[m+2]=i; c[m+2]=j_pp; d[m+2]= 1/3
            m += 3
            if classified_grid[k+3,l]==2:
                j_ppp = np.where((U_ind[:,0]==k+3) & (U_ind[:,1]==l))[0][0]
                r[m]=i; c[m]=j_ppp; d[m]=-1/12
                m+=1
        elif neighbours[3]==1:
            r[m]=i; c[m]=i; d[m]= -5/3
            j_m = np.where((U_ind[:,0]==k-1) & (U_ind[:,1]==l))[0][0]
            r[m+1]=i; c[m+1]=j_m; d[m+1]= 1/2
            j_mm = np.where((U_ind[:,0]==k-2) & (U_ind[:,1]==l))[0][0]
            r[m+2]=i; c[m+2]=j_mm; d[m+2]= 1/3
            m += 3
            if classified_grid[k-3,l]==2:
                j_mmm = np.where((U_ind[:,0]==k-3) & (U_ind[:,1]==l))[0][0]
                r[m]=i; c[m]=j_mmm; d[m]=-1/12
                m+=1
        else:
            r[m]=i; c[m]=i; d[m]= -5/2
            j_p = np.where((U_ind[:,0]==k+1) & (U_ind[:,1]==l))[0][0]
            r[m+1]=i; c[m+1]=j_p; d[m+1]= 4/3
            j_m = np.where((U_ind[:,0]==k-1) & (U_ind[:,1]==l))[0][0]
            r[m+2]=i; c[m+2]=j_m; d[m+2]= 4/3
            m += 3
            if classified_grid[k-2,l]==2:
                j_pp = np.where((U_ind[:,0]==k-2) & (U_ind[:,1]==l))[0][0]
                r[m]=i; c[m]=j_pp; d[m]=-1/12;
                m+=1
            if classified_grid[k+2,l]==2:
                j_mm = np.where((U_ind[:,0]==k+2) & (U_ind[:,1]==l))[0][0]
                r[m]=i; c[m]=j_mm; d[m]=-1/12;
                m+=1
    if (m > space*n):
        print('Error, make more room')
    return r[:m],c[:m],d[:m]

def make_matrix_HO(classified_grid,delta):
    U_ind = np.argwhere(classified_grid==2).astype(np.uint32)
    n = U_ind.shape[0]
    r,c,d = COO_matrix_HO(U_ind,classified_grid)
    d = -d/(delta**2)
    B = sparse.coo_matrix((d,(r,c)),shape=(n,n),dtype=np.float32).tocsr()
    return B, U_ind

#plot_mode(3,4,0,c_file='Data/c_grid34.npz',func=make_matrix_HO)
