import numpy as np
import numba as nb
import cmath

from fractal import *

# ----- Ray Method
@nb.njit
def on_boundary_ray(px,py,fractal,x_axis=True):
    if x_axis:
        frac_points = fractal[(fractal[:,1]==py)!=0,0]
        if frac_points.size > 0:
            frac_points = np.sort(frac_points)
            if px < frac_points[0] or px > frac_points[-1]:
                return 0
            index = np.argwhere(px <= frac_points)[0,0]
            i = np.where((fractal[:,0]==frac_points[index]) & (fractal[:,1]==py))[0]
            j = np.where((fractal[:,0]==frac_points[index-1]) & (fractal[:,1]==py))[0]
            if (px > frac_points[index-1] and ((np.abs(i-j) == 1) or (i==len(fractal)-1) and j==0)) or (px==frac_points[index]):
                #the fractal 'completes' its round along the x-axis due to how it is built up. Keep this in mind!
                return 1
    else:
        frac_points = fractal[(fractal[:,0]==px)!=0,1]
        if frac_points.size > 0:
            frac_points = np.sort(frac_points)
            if py < frac_points[0] or py > frac_points[-1]:
                return 0
            index = np.argwhere(py <= frac_points)[0,0]
            i = np.where((fractal[:,0]==px) & (fractal[:,1]==frac_points[index]))[0]
            j = np.where((fractal[:,0]==px) & (fractal[:,1]==frac_points[index-1]))[0]
            if (py > frac_points[index-1] and np.abs(i-j) == 1)  or (py==frac_points[index]):
                return 1    
    return 0.5 #value to continue

@nb.njit
def orientation(p,q,r):
    val = ((q[1]-p[1])*(r[0]-q[0])) - ((q[0]-p[0])*(r[1]-q[1]))
    if val>0.0:
        val = 1.0
    else:
        val = -1.0
    return val

@nb.njit
def intersect(p1,p2,q1,q2):
    boo = False
    o_1 = orientation(p1,p2,q1)
    o_2 = orientation(p1,p2,q2)
    o_3 = orientation(q1,q2,p1)
    o_4 = orientation(q1,q2,p2)
    if (o_1 != o_2) and (o_3 != o_4):
        boo = True
    return boo

@nb.njit
def classify_grid_point_ray(point,fractal,delta):
    py = point[1]
    px = point[0]
    
    on_x_boundary = on_boundary_ray(px,py,fractal)
    if on_x_boundary == 0:
        return 0
    elif on_x_boundary == 1:
        return 1
                
    on_y_boundary = on_boundary_ray(px,py,fractal,False)
    if on_y_boundary == 0:
        return 0
    elif on_y_boundary == 1:
        return 1
    
    px += delta/2
    y_max = np.amax(fractal[:,1]) 
    p2 = np.array([px,y_max+delta],dtype=np.float64)
    p1 = np.array([px,py],dtype=np.float64)

    count = 0
    for i in range(len(fractal[:,0])-1):
        if intersect(p1,p2,fractal[i,:],fractal[i+1,:]):
            count += 1
    count += intersect(p1,p2,fractal[-1,:],fractal[0,:])
    if count % 2 == 0:
        return 0
    else:
        return 2
    
# --- Complex Integration Method
@nb.njit
def within(p,q,r):
    return p <= q <= r or r <= q <= p

@nb.njit
def point_on_boundary(point,P0,P1):
    p0 = [P0[0]-point[0],P0[1]-point[1]]
    p1 = [P1[0]-point[0],P1[1]-point[1]]

    det = p0[0]*p1[1] - p1[0]*p0[1]
    if P0[0] != P1[0]:
        return det == 0 and within(P0[0],point[0],P1[0])
    else:
        return det == 0 and within(P0[1],point[1],P1[1])

@nb.njit
def classify_grid_point_complex_int(point,fractal):
    val = 0
    sum = np.complex64(0)    
    num = np.complex64(0)
    denom = np.complex64(0)
    for i in range(-1,len(fractal)-1):
        p_now, p_next = fractal[i,:],fractal[i+1,:]
        if point_on_boundary(point,p_now,p_next):
            val = 1
            return val
        num = complex((p_next[0]-point[0]),(p_next[1]-point[1]))
        denom = complex((p_now[0]-point[0]),(p_now[1]-point[1]))
        sum += cmath.log(num/denom)   
    
    if np.abs(sum) > 1:
        val = 2
    return val

# ---- Utils ---
def initialize_grid(l_max): #assumes initialize_square placed at origin
    L = 1.0
    delta = L*4**(-l_max)
    additional_dist = 0
    for i in range(l_max):
        additional_dist += L*4**(-(i+1))
    xv,yv = np.meshgrid(np.arange(-additional_dist,L+additional_dist+delta,delta),np.arange(-additional_dist,L+additional_dist+delta,delta),indexing='xy')

    return xv,yv #can return Nx2 matrix (x,y)-points with; grid = np.column_stack((xv.flatten(),yv.flatten()))

@nb.njit
def classify_grid(frac,xv,yv):
    N = xv.shape[0]
    delta = abs(xv[0,0]-xv[0,1])
    classified_grid = np.zeros((N,N),dtype=np.uint8)
    point = np.zeros(2,dtype=np.float64)
    for i in range(N):
        for j in range(N):
            point[0], point[1] = xv[i,j], yv[i,j]
            classified_grid[i,j] = classify_grid_point_ray(point,frac,delta)
    return classified_grid

def save_classifier(l,l_max):
    frac = Koch_fractal(l)
    xv,yv = initialize_grid(l_max)
    classified_grid = classify_grid(frac,xv,yv)
    filename = 'c_grid' + str(l) + str(l_max) + '.npz'
    np.savez(filename,classified_grid)