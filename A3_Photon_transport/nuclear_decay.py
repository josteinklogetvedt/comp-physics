
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
lam = 0.3

def MC_decay(t,N_0,lambda_arr):
    N = np.zeros((len(t),len(lambda_arr)+1),dtype=np.int32)
    N[:,0] = N_0*np.ones(len(t))
    dt = t[1]-t[0]
    
    P_decay = 1-np.exp(-lambda_arr*dt)
    for i in range(1,len(t)):
        
        for j in range(len(lambda_arr)-1,-1,-1):
            if N[i,j] > 0:
                
                random_numbers = np.random.rand(N[i,j])
                number_of_decays = np.sum(random_numbers <= P_decay[j])
            
                if number_of_decays > 0:
                    N[i:,j] = N[i,j] - number_of_decays
                    N[i:,j+1] = N[i,j+1] + number_of_decays

    return N

def N_analytic(t,N_0,lam):
    return N_0*np.exp(-lam*t)

# --- Plotting -------

def plot_N_given_N_0(savefile=False):
    N_0 = 200
    T = 2*(1/lam)
    points_1 = 2*10**4
    points_2 = 100
    
    lambda_arr = np.asarray([lam])
    t_1 = np.linspace(0,T,points_1)
    t_2 = np.linspace(0,T,points_2)
    N_1 = MC_decay(t_1,N_0,lambda_arr)
    N_2 = MC_decay(t_2,N_0,lambda_arr)
    
    plt.plot(t_1[::20],N_1[::20,0],color='deepskyblue',label=r'$N_{MC} \: \: n=$'+str(points_1))
    plt.plot(t_2,N_2[:,0],color='blue',label=r'$N_{MC} \: \: n=$'+str(points_2))
    plt.plot(t_2,N_analytic(t_2,N_0,lam),color='red',linestyle='dashed',label=r'$N(t)$'+' analytical')
    plt.xlabel(r'$t \: (s)$')
    plt.ylabel(r'$N(t)$')
    plt.legend()
    plt.xlim(0,t_1[-1])
    plt.title(r'$N_0=$'+str(N_0))
    if savefile:
        plt.savefig(savefile)
    plt.show()

def plot_N_given_points(savefile=False):
    T = 2*(1/lam)
    points = 1000
    N_0_1 = 200
    N_0_2 = 1000
    N_0_3 = 10000
        
    lambda_arr = np.asarray([lam])
    t = np.linspace(0,T,points)
    N_1 = MC_decay(t,N_0_1,lambda_arr)
    N_2 = MC_decay(t,N_0_2,lambda_arr)
    N_3 = MC_decay(t,N_0_3,lambda_arr)
    
    plt.plot(t,N_1[:,0],color='deepskyblue',label=r'$N_0 =$'+str(N_0_1))
    plt.plot(t,N_2[:,0],color='blue',label=r'$N_0=$'+str(N_0_2))
    plt.plot(t,N_3[:,0],color='darkblue',label=r'$N_0=$'+str(N_0_3))
    plt.xlabel(r'$t \: (s)$')
    plt.ylabel(r'$N(t)$')
    plt.legend()
    plt.yscale('log')
    plt.xlim(0,t[-1])
    plt.title(r'$n=$'+str(points))
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
#plot_N_given_N_0() 
#plot_N_given_points()

# ---- Plot error ----

def plot_error_given_N_0(savefile=False):
    T = 2*(1/lam)
    tryouts = 100
    N_0 = 200
    points = np.logspace(2,5,10,dtype=np.int32)*2
    
    lambda_arr = np.asarray([lam])
    err = np.zeros(len(points))
    for i in range(len(points)):
        err_mean = np.zeros(tryouts)
        t = np.linspace(0,T,points[i])
        N_sol = N_analytic(t,N_0,lam)
        
        for j in range(tryouts):
            N = MC_decay(t,N_0,lambda_arr)
            err_mean[j] = np.linalg.norm(N_sol-N[:,0])/np.linalg.norm(N_sol)
            
        err[i] = np.mean(err_mean)
        
    plt.plot(points, err, color='blue', linestyle='dashed',marker='o',label=r'$\frac{||N-N_{MC}||}{N}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$n$')
    plt.ylabel('Error')
    plt.title(r'$N_0=$'+str(N_0))
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()

def plot_error_given_points(savefile=False):
    T = 2*(1/lam)
    tryouts = 100
    points = 1000
    N_0 = np.logspace(2,5,10)*2
    
    lambda_arr = np.asarray([lam])
    t = np.linspace(0,T,points)
    err = np.zeros(len(N_0))
    for i in range(len(N_0)):
        err_mean = np.zeros(tryouts)
        N_sol = N_analytic(t,N_0[i],lam)
        
        for j in range(tryouts):
            N = MC_decay(t,N_0[i],lambda_arr)
            err_mean[j] = np.linalg.norm(N_sol-N[:,0])/np.linalg.norm(N_sol)
            
        err[i] = np.mean(err_mean)
        
    plt.plot(N_0, err, color='blue', linestyle='dashed',marker='o',label=r'$\frac{||N-N_{MC}||}{N}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'$N_0$')
    plt.ylabel('Error')
    plt.title(r'$n=$'+str(points))
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
#plot_error_given_N_0()
#plot_error_given_points()

def plot_error_asaf_time(savefile=False):
    N_0 = 2000
    points = 500
    end_times = np.linspace(0.1,6,20)*(np.log(2)/lam) 
    tryouts = 100
    
    error = np.zeros(len(end_times))
    lambda_arr = np.asarray([lam])
    for i,T in enumerate(end_times):
        t = np.linspace(0,T,points)
        N_sol = N_analytic(T,N_0,lam)
        err_mean = np.zeros(tryouts)
        
        for j in range(tryouts):
            N = MC_decay(t,N_0,lambda_arr)
            err_mean[j] = np.abs(N_sol-N[-1,0])/N_sol
            
        error[i] = np.mean(err_mean)
        
    plt.plot(end_times,error,linestyle='dashed',marker='o',color='blue',label=r'$\frac{|N(T)-N_{MC}|}{|N(t)|}$')
    plt.xlabel(r'$T \: (s)$')
    plt.ylabel('Error')
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()

#plot_error_asaf_time()

def plot_N_distr(savefile=False):
    T = (1/lam)
    N_0 = 5000
    points = 500
    tryouts = 5000
    
    lambda_arr = np.asarray([lam])
    t = np.linspace(0,T,points)
    N_sol = N_analytic(T,N_0,lam)
    N_distr = np.zeros(tryouts)
    for i in range(tryouts):
        N = MC_decay(t,N_0,lambda_arr)
        N_distr[i] = N[-1,0]
    
    mean = np.mean(N_distr)
    std = np.std(N_distr)
    deviation = np.abs(mean-N_sol)/N_sol
    print(f'Mean value N_distr: {mean} with deviation {deviation}')
    print(f'Standard deviation: {std}')
    
    counts, bins, bars = plt.hist(N_distr,bins=50,density=True,alpha=0.8,label=r'$N_{MC}$')
    
    x = np.linspace(np.amin(N_distr),np.amax(N_distr),100)
    plt.plot(np.ones(100)*N_sol,np.linspace(0,np.amax(counts),100),linewidth=2.5,color='red',label=r'$N(T)$')
    plt.plot(x,np.exp(-np.square(x-N_sol)/(2*(std)**2))/(np.sqrt(2*np.pi*(std)**2)),linestyle='dashed',color='green',label='Gaussian')
    plt.xlabel(r'$N$')
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()

#plot_N_distr()


# --- Secular Equilibrium -------
def N_2_analytic(t,N_0,lambda_arr):
    return ((lambda_arr[0]*N_0)/(lambda_arr[1]-lambda_arr[0]))*(np.exp(-lambda_arr[0]*t)-np.exp(-lambda_arr[1]*t))

def plot_all_Ns(savefile=False):
    lam_1 = lam
    lam_2 = 3*lam
    T = 2*(1/lam_1)
    points = 4000
    N_0 = 500
    
    t = np.linspace(0,T,points)
    lambda_arr = np.asarray([lam_1,lam_2])
    N = MC_decay(t,N_0,lambda_arr)

    plt.plot(t,N[:,0],color='blue',label=r'$N_{1} \: \: \lambda_1=$'+str(round(lam_1,1)))
    plt.plot(t,N[:,1],color='green',label=r'$N_{2} \: \: \lambda_2=$'+str(round(lam_2,1)))
    plt.plot(t,N_analytic(t,N_0,lam),color='cornflowerblue',linestyle='dashed',label=r'$N_1(t)$'+' analytical')
    plt.plot(t,N_2_analytic(t,N_0,lambda_arr),color='springgreen',linestyle='dashed',label=r'$N_2(t)$'+' analytical')
    
    plt.xlabel(r'$t \: (s)$')
    plt.ylabel(r'$N(t)$')
    plt.legend()
    plt.xlim(0,t[-1])
    plt.title(r'$N_0=$'+str(N_0))
    if savefile:
        plt.savefig(savefile)
    plt.show()

def plot_sec_equil(savefile=False):
    lam_1 = lam
    lam_2 = 200*lam

    T = 20*(np.log(2)/lam_2)
    points = 4000
    N_0 = 15000
    t = np.linspace(0,T,points)
    lambda_arr = np.asarray([lam_1,lam_2])
    N = MC_decay(t,N_0,lambda_arr)

    plt.plot(t,N[:,1],color='green',label=r'$N_{2} \: \: \lambda_2=$'+str(round(lam_2,1)))
    plt.plot(t,N_2_analytic(t,N_0,lambda_arr),color='springgreen',linestyle='dashed',label=r'$N_2(t)$'+' analytical')
    plt.plot(t,np.ones_like(t)*N_0*(lam_1/lam_2),linestyle='dashdot',color='black',label='Secular equilibrium')
    
    plt.xlabel(r'$t \: (s)$')
    plt.ylabel(r'$N(t)$')
    plt.legend()
    plt.xlim(0,t[-1])
    plt.ylim(0,N_0*(lam_1/lam_2)+80)
    plt.title(r'$N_0=$'+str(N_0))
    if savefile:
        plt.savefig(savefile)
    plt.show()

#plot_all_Ns() 
#plot_sec_equil()