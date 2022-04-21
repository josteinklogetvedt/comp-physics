import numpy as np
import numba as nb

@nb.njit
def generator(s_points):
    end_points = np.zeros((8,2),dtype=np.float32)
    end_points[0,:] = s_points[0,:]
    s = np.abs(np.sum(s_points[0,:]-s_points[1,:]))/4
    
    S = np.array([[0.0,s],[-s,s],[-s,2*s],[0.0,2*s],[s,2*s],[s,3*s],[0.0,3*s]],dtype=np.float32)
    S_rev = np.ascontiguousarray(S[:,::-1])
    mat = np.array([[1.0,0.0],[0.0,-1.0]],dtype=np.float32)
    
    if s_points[0,0]==s_points[1,0] and s_points[0,1] < s_points[1,1]:
        end_points[1:,:] = s_points[0,:] + S
    elif s_points[0,0]==s_points[1,0] and s_points[0,1] > s_points[1,1]:
        end_points[1:,:] = (s_points[1,:] + S)[::-1]
    elif s_points[0,1]==s_points[1,1] and s_points[0,0] < s_points[1,0]:
        np.dot(S_rev,mat,S)
        end_points[1:,:] = s_points[0,:] + S
    elif s_points[0,1]==s_points[1,1] and s_points[0,0] > s_points[1,0]:
        np.dot(S_rev,mat,S)
        end_points[1:,:] = (s_points[1,:] + S)[::-1]
    return end_points

#@nb.njit  #for large l>8
def fractalize(points):
    n_rows = len(points)
    res = np.zeros((n_rows*8,2),dtype=np.float32)

    index = 0
    for i in range(n_rows-1):
        res[index:index+8,:] = generator(points[i:i+2,:])
        index += 8
    res[index:index+8,:] = generator(np.vstack((points[-1],points[0])))
    return res

@nb.njit
def initialize_square():
    L = 1.0
    square = np.zeros((4,2),dtype=np.float32)
    square[1,1] = square[2,0] = square[2,1] = square[3,0] = L
    return square

#@nb.njit  #for large l>8
def Koch_fractal(l):
    end = np.zeros((4*8**l,2),dtype=np.float32)
    end[0:4,:] = initialize_square()
    
    for i in range(l):
        end[:4*8**(i+1),:] = fractalize(end[:4*8**(i),:])
    return end