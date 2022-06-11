from spin_lattice import *

# Code to make plots for tasks g), h) and i)

def plot_GS_lattice(T_end,N,B,savefile=False):  # Plots magnetization when T=0
    S_0 = np.random.uniform(low=-1,high=1,size=(N*N,3))  # randomly initialize spins
    Th = 0.0   # no thermal energy
    
    dt = 0.001*ps
    num = int(T_end/dt)
    t = np.linspace(0,T_end,num)
    
    S = time_solver(S_0,t,B,Th)
    M = (1/N**2)*np.sum(S[:,:,2],axis=1)
    
    plt.plot(t/ps,M,color='blue',label=r'$M(T,t)$')
    plt.xlabel(r'$t$'+' (ps)')
    plt.ylabel(r'$M$')
    plt.xlim(-0.3,T_end/ps + 0.03)
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()

def plot_magnetization(T_end,t_eq,N,B,Th,savefile=False): # Plots M(T,t) together with M_ave, averaged from t_eq to T_end
    S_0 = np.zeros((N*N,3))
    S_0[:,2] = np.ones(N*N)

    dt = 0.001*ps
    num = int(T_end/dt)
    t = np.linspace(0,T_end,num)
    idx = np.abs(t-t_eq).argmin()
    
    S = time_solver(S_0,t,B,Th)
    M = (1/N**2)*np.sum(S[:,:,2],axis=1)
    M_ave = np.average(M[idx:])
    M_std = np.std(M[idx:])
    print('Averag M is;',M_ave,'with standard deviation',M_std)
    
    plt.plot(t/ps,M,color='blue',label=r'$M(T,t)$')
    grid = np.linspace(0,t[-1],40)
    plt.plot(grid/ps,np.ones_like(grid)*M_ave,color='black',linewidth=3,label=r'$M(T)$')
    plt.plot(grid/ps,np.ones_like(grid)*(M_ave-M_std),color='black',linewidth=1.5,linestyle='dashed')
    plt.plot(grid/ps,np.ones_like(grid)*(M_ave+M_std),color='black',linewidth=1.5,linestyle='dashed')
    plt.xlabel(r'$t$'+' (ps)')
    plt.ylabel(r'$M$')
    plt.title(r'$T=$'+str(round(Th*J/k_b,2))+' K')
    plt.xlim(-0.3,T_end/ps+0.3)
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()

def plot_phase_transition(Th_arr,file,savefile=False): # Plots M(T) for diferent T's. Scaled T axis (T/T_c).
    M = np.load(file)['arr_0']
    M_std = np.load(file)['arr_1']
    
    M_upper = M + M_std
    M_lower = M - M_std

    T = J*Th_arr/k_b
    upper_limit = np.argwhere(M-M_std < 0)[0]
    T_c = (T[upper_limit]+T[upper_limit-1])/2
    print('Critical temperature is;', T_c)
    
    plt.scatter(T/T_c,M,color='blue',s=4)
    plt.fill_between(T/T_c,M_upper,M_lower,alpha=0.1,color='blue')
    
    plt.xlabel(r'$T/T_c$')
    plt.ylabel(r'$M(T)$')
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
# Start with B=0.5 J/mu, choose T_end = 20ps, t_eq = 10ps, run till Th=23 (relative thermal energy).
N = 50
B = 0.5*J/mu
T_end = 20*ps
t_eq = 10*ps
Th_arr = np.linspace(0,23,70)

#plot_GS_lattice(T_end,N,B)  # task g) 
#plot_magnetization(T_end,t_eq,N,B,Th=0.05)  # Low Th < 0.1, task g) 
#phase_diagram(T_end,t_eq,N,B,Th_arr,savefile='B_0.5_Th_23_70_w_dz.npz') # save to file
#plot_phase_transition(Th_arr,'Data/B_0.5_Th_23_70_w_dz.npz') # task h)

# -------------------
# Task i) below, aggregate similar results for different B's and compare

# Now B = 0.25* J/mu, choose T_end = 25ps, t_eq = 15ps, run till Th=15 (relative thermal energy).
N = 50
B = 0.25*J/mu
T_end = 25*ps
t_eq = 15*ps
Th_arr = np.linspace(0,15,60)
#phase_diagram(T_end,t_eq,N,B,Th_arr,savefile='B_0.25_Th_15_60_w_dz.npz') # save to file


# Now, B=0, choose T_end = 30ps, t_eq = 20ps, run till Th=4.5 (relative thermal energy).
N = 50
B = 0.0
T_end = 30*ps
t_eq = 20*ps
Th_arr = np.linspace(0,4.5,60)  
#phase_diagram(T_end,t_eq,N,B,Th_arr,savefile='B_0.0_Th_4.5_60_w_dz.npz') # save to file



def plot_many_PT(file_1,file_2,file_3,savefile=False): # Plot phase-diagrams for all B's. Use regular T-axis
    M_low = np.load(file_1)['arr_0']
    M_low_std = np.load(file_1)['arr_1']
    M_middle = np.load(file_2)['arr_0']
    M_middle_std = np.load(file_2)['arr_1']
    M_high = np.load(file_3)['arr_0']
    M_high_std = np.load(file_3)['arr_1']
    
    plt.scatter(np.linspace(0,4.5,60)*J/k_b,M_low,color='blue',s=4,label=r'$B_0 = 0$')
    plt.fill_between(np.linspace(0,4.5,60)*J/k_b,M_low + M_low_std,M_low - M_low_std,alpha=0.1,color='blue')
    plt.scatter(np.linspace(0,15,60)*J/k_b,M_middle,color='red',s=4,label=r'$B_0 =0.25 J/\mu_s$')
    plt.fill_between(np.linspace(0,15,60)*J/k_b,M_middle + M_middle_std,M_middle - M_middle_std,alpha=0.1,color='red')
    plt.scatter(np.linspace(0,23,70)*J/k_b,M_high,color='green',s=4,label=r'$B_0 =0.5 J/\mu_s$')
    plt.fill_between(np.linspace(0,23,70)*J/k_b,M_high + M_high_std,M_high - M_high_std,alpha=0.1,color='green')
    
    plt.xlabel(r'$T$'+' (K)')
    plt.ylabel(r'$M(T)$')
    plt.xlim(-1,23*J/k_b+1)
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()
 
#plot_many_PT('Data/B_0.0_Th_4.5_60_w_dz.npz','Data/B_0.25_Th_15_60_w_dz.npz','Data/B_0.5_Th_23_70_w_dz.npz') # task i)

   
def plot_many_PT_normed(file_1,file_2,file_3,savefile=False): # Plot phase-diagrams for all B's. Use scaled T/T_c-axis
    M_low = np.load(file_1)['arr_0']
    M_low_std = np.load(file_1)['arr_1']
    M_middle = np.load(file_2)['arr_0']
    M_middle_std = np.load(file_2)['arr_1']
    M_high = np.load(file_3)['arr_0']
    M_high_std = np.load(file_3)['arr_1']
    
    T_low = np.linspace(0,4.5,60)*J/k_b
    T_middle = np.linspace(0,15,60)*J/k_b
    T_high = np.linspace(0,23,70)*J/k_b
    
    T_c_low = T_low[np.argwhere(M_low-M_low_std < 0)[0]]
    T_c_middle = T_middle[np.argwhere(M_middle-M_middle_std < 0)[0]]
    T_c_high = T_high[np.argwhere(M_high-M_high_std < 0)[0]]
    
    print('Critical temperatures;',T_c_low,T_c_middle,T_c_high)
    
    plt.scatter(T_low/T_c_low,M_low,color='blue',s=4,label=r'$B_0 = 0$')
    plt.fill_between(T_low/T_c_low,M_low + M_low_std,M_low - M_low_std,alpha=0.1,color='blue')
    plt.scatter(T_middle/T_c_middle,M_middle,color='red',s=4,label=r'$B_0 =0.25 J/\mu_s$')
    plt.fill_between(T_middle/T_c_middle,M_middle + M_middle_std,M_middle - M_middle_std,alpha=0.1,color='red')
    plt.scatter(T_high/T_c_high,M_high,color='green',s=4,label=r'$B_0 =0.5 J/\mu_s$')
    plt.fill_between(T_high/T_c_high,M_high + M_high_std,M_high - M_high_std,alpha=0.1,color='green')
    
    plt.xlabel(r'$T/T_c$')
    plt.ylabel(r'$M(T)$')
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
#plot_many_PT_normed('Data/B_0.0_Th_4.5_60_w_dz.npz','Data/B_0.25_Th_15_60_w_dz.npz','Data/B_0.5_Th_23_70_w_dz.npz') # task i)


def plot_PT_no_dz(f1,f1_d,Th_arr,savefile=False): # Plot phase-diagrams for a particular B, with and without the d_z term
    M_1 = np.load(f1)['arr_0']
    M_1_std = np.load(f1)['arr_1']
    M_1d = np.load(f1_d)['arr_0']
    M_1d_std = np.load(f1_d)['arr_1']
    
    T_c_1 = Th_arr[np.argwhere(M_1-M_1_std < 0)[0]]
    T_c_1d = Th_arr[np.argwhere(M_1d-M_1d_std < 0)[0]]

    print('Critical temperatures;',T_c_1,T_c_1d)
    
    plt.scatter(Th_arr/T_c_1,M_1,color='red',s=5,label=r'$d_z=0$')
    plt.fill_between(Th_arr/T_c_1,M_1 + M_1_std,M_1 - M_1_std,alpha=0.1,color='red')
    plt.scatter(Th_arr/T_c_1d,M_1d,color='blue',s=5,label=r'$d_z>0$')
    plt.fill_between(Th_arr/T_c_1d,M_1d + M_1d_std,M_1d - M_1d_std,alpha=0.1,color='blue')
    
    plt.xlabel(r'$T/T_c$')
    plt.ylabel(r'$M(T)$')
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show() 
    
T_1 = np.linspace(0,4.5,60)*J/k_b
T_2 = np.linspace(0,23,70)*J/k_b
#plot_PT_no_dz('Data/B_0.0_Th_4.5_60.npz','Data/B_0.0_Th_4.5_60_w_dz.npz',T_1)  # task i)
#plot_PT_no_dz('Data/B_0.5_Th_23_70.npz','Data/B_0.5_Th_23_70_w_dz.npz',T_2)    # i)
