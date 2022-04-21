import numpy as np
import matplotlib.pyplot as plt
from numpy import random

def middle_square_method(seed,length_sequence): #seed = non-negative integer, 
    n = len(str(seed))
    assert(n % 2==0) # must have even n
    sequence = np.zeros(length_sequence)
    
    for i in range(length_sequence):
        s_squared = seed**2
        s_squared_string = str(s_squared)
        n_new = len(s_squared_string)
        while n_new < 2*n:
            s_squared_string = '0' + s_squared_string
            n_new += 1

        s_new_string = s_squared_string[n//2:2*n-n//2]
        seed = int(s_new_string)
        sequence[i]=seed
        
    return sequence

def lin_cong_method(seed,a,c,m,length_sequence):
    assert(0 < a < m)
    assert(0 <= c < m)
    sequence = []
    filled_in = 0
    sequence = np.zeros(length_sequence)
    sequence[0] = (a*seed + c) % m
    for i in range(length_sequence-1):
        sequence[i+1] = (a*sequence[i] + c) % m
        
    return sequence

# --- Plotting ------
s_1 = 67348210   # MS
s_2 = 67348212  
m_1 = 85726   # LCG
m_2 = 85722
a = 1234
c = 342
X_0 = 8643

def plot_imshow(N,save=False):
    length = N**2
    
    seq_MS_1 = middle_square_method(s_1,length)
    seq_MS_2 = middle_square_method(s_2,length)
    seq_LCG_1 = lin_cong_method(X_0,a,c,m_1,length)
    seq_LCG_2 = lin_cong_method(X_0,a,c,m_2,length)

    seq_MS_1 = seq_MS_1/np.amax(seq_MS_1)
    seq_MS_2 = seq_MS_2/np.amax(seq_MS_2)
    seq_LCG_1 = seq_LCG_1/np.amax(seq_LCG_1)
    seq_LCG_2 = seq_LCG_2/np.amax(seq_LCG_2)
    
    plt.imshow(seq_MS_1.reshape((N,N)),cmap='gray')
    if save:
        plt.savefig('MS_1.png')
    plt.show()
    plt.imshow(seq_MS_2.reshape((N,N)),cmap='gray')
    if save:
        plt.savefig('MS_2.png')
    plt.show()
    plt.imshow(seq_LCG_1.reshape((N,N)),cmap='gray')
    if save:
        plt.savefig('LCG_1.png')
    plt.show()
    plt.imshow(seq_LCG_2.reshape((N,N)),cmap='gray')
    if save:
        plt.savefig('LCG_2.png')
    plt.show()
    
#plot_imshow(100)

def plot_points_3d(sequence,savefile=False):
    seq = np.copy(sequence)
    seq = seq/np.amax(seq)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    x = seq[::3]; y= seq[1::3]; z = seq[2::3]
    
    ax.scatter3D(x,y,z,c='blue',s=3.4)
    ax.view_init(elev=25,azim=-73)
    if savefile:
        plt.savefig(savefile)
    plt.show()

#points = 2*10**4
#seq_MS = middle_square_method(s_2,points*3)
#seq_LCG = lin_cong_method(X_0,a,c,m_1,points*3)
#plot_points_3d(seq_MS)
#plot_points_3d(seq_LCG)

def plot_uniform_distr(MS,LCG,savefile=False):
    seq_MS = np.copy(MS)
    seq_LCG = np.copy(LCG)
    
    seq_MS = seq_MS/np.amax(seq_MS)
    seq_LCG = seq_LCG/np.amax(seq_LCG)
    bins = 100
    size = 6
    
    c_MS, bins_MS = np.histogram(seq_MS,bins=bins)
    c_LCG, bins_LCG = np.histogram(seq_LCG,bins=bins)
    c_MS_scaled = c_MS/np.mean(c_MS)
    c_LCG_scaled = c_LCG/np.mean(c_LCG)
    print(f'Mean MS: {np.mean(c_MS_scaled)} with std {np.std(c_MS_scaled)}')
    print(f'Mean LCG: {np.mean(c_LCG_scaled)} with std {np.std(c_LCG_scaled)}')
    
    plt.scatter(bins_MS[:-1],c_MS_scaled,label='MS',color='blue',s=size)
    plt.scatter(bins_LCG[:-1],c_LCG_scaled,label='LCG',color='green',s=size)
    
    np.random.seed(42)
    seq_numpy = np.random.rand(len(seq_MS))
    c_numpy, bins_numpy = np.histogram(seq_numpy,bins=bins)
    c_numpy_scaled = c_numpy/np.mean(c_numpy)
    plt.scatter(bins_numpy[:-1],c_numpy_scaled,label='Numpy',color='red',s=size)
    print(f'Mean Numpy: {np.mean(c_numpy_scaled)} with std {np.std(c_numpy_scaled)}')
    
    plt.legend()
    plt.xlim(0-0.02,1+0.02)
    plt.ylim(0.3,1.4)
    if savefile:
        plt.savefig(savefile)
    plt.show() 

#length = 1*10**4
#seq_MS = middle_square_method(s_1,length)
#seq_LCG = lin_cong_method(X_0,a,c,m_1,length)
#plot_uniform_distr(seq_MS,seq_LCG)