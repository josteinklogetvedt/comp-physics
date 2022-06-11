import matplotlib.pyplot as plt

from solver import *

#Constants
dU = 80  #eV
dt = 2*10**-5
r_1 = r
r_2 = 3*r

def plot_trajectory_wflash(tau_arr,cycles,r,dt): # high freq; tau = 0.04, intermediate; tau = 1; low frequency; tau = 25
    assert(time_criterion(dt,dU,r))
    
    x_0 = np.asarray([0.0])
    fig, axs = plt.subplots(len(tau_arr))
    
    for i in range(len(tau_arr)):
        tau = tau_arr[i]      
        T_end = cycles*tau    # stop simulation at the end of an cycle
        t = np.arange(0,T_end+dt,dt)
        f = 1/tau
        
        print('f=',f)
        print('Number of points;',len(t))

        x, pot = solver(t,x_0,tau,dU,r,flashing=True)

        t_ax = np.arange(0,T_end+dt,tau)
        x_red = x[:,0]*micrometer

        axs[i].plot(t,x_red,color='blue',label=r'$f=$'+str(round(f,2)))
        axs[i].set_xticks(t_ax,["{:.1f}".format(t_ax[i]) for i in range(len(t_ax))])

        height = np.amax(x_red)-np.amin(x_red)
        axs[i].set_ylim(np.amin(x_red)-height/10,np.amax(x_red)+height/10)
        axs[i].set_xlim(t[0]-dt,t[-1]+5*dt)
        
        for j in range(1,len(t_ax)):
            idx = np.abs(t - t_ax[j]).argmin()
            x_val = x_red[idx]
            axs[i].plot(np.ones(10)*t_ax[j],np.linspace(x_val-height/10,x_val+height/10,10),linewidth=1.1,color='black')
        axs[i].set_ylabel(r'$x \: (\mu m)$')
        axs[i].legend()
    axs[len(tau_arr)-1].set_xlabel(r'$t \: (s)$')
    fig.tight_layout()
    #plt.savefig('traject_w_flash.pdf')
    plt.show()
    
cycles = 10 
tau_arr = np.asarray([0.04,1,25])
#plot_trajectory_wflash(tau_arr,cycles,r,dt)   # <-- run this for plot. r = r_1 here


# ----------- Velocity calculations--------------------
@nb.njit
def calc_ave_velocity(tau_arr,T_end,datapoints,r,dt):
    cycles = T_end//tau_arr
    
    ave_velocity = np.zeros(len(tau_arr))
    std_velocity = np.zeros(len(tau_arr))
    x_0 = np.zeros(datapoints)    

    for i in range(len(tau_arr)):
        t = np.arange(0,cycles[i]*tau_arr[i],dt)      # at the end of a cycle
        x = solver_only_last(t,x_0,tau_arr[i],dU,r)
        v = x/t[-1]
        ave_velocity[i] = np.mean(v)
        std_velocity[i] = np.std(v) 
         
    return ave_velocity, std_velocity

def find_velocity(tau_arr,T_end,datapoints,r_arr,dt,savefile=False): #just a wrapper in order to to save file
    ave_v = np.zeros((len(tau_arr),len(r_arr)))
    std_v = np.zeros((len(tau_arr),len(r_arr)))
    
    for i in range(len(r_arr)):
        assert(time_criterion(dt,dU,r_arr[i]))
        ave_v[:,i], std_v[:,i] = calc_ave_velocity(tau_arr,T_end,datapoints,r_arr[i],dt)
        
        if savefile:
            k = round(r_arr[i]*nanometer)
            np.savez('Data/ave_v_'+str(k),ave_v[:,i])
            np.savez('Data/std_v_'+str(k),std_v[:,i])
    return ave_v, std_v

r_arr = np.asarray([r_1,r_2])
tau_arr = np.linspace(0.05,2,100)  # 1.5 hours to run
T_end = 100        #use these values, maybe higher datapoints if neccesary
datapoints = 120
#a_v,s_v = find_velocity(tau_arr,T_end,datapoints,r_arr,dt,savefile=True)  # <-- run this, saves velocities in memory


from scipy.signal import savgol_filter  # employ savgol filter to the errorbars for prettier plot

def plot_velocities(tau_arr,ave_v1,std_v1,ave_v2,std_v2,estimation=False):
    plt.figure(figsize=(9,5))
    
    plt.scatter(tau_arr,ave_v1,c='blue',s=2.5,label='12'+'nm')
    plt.scatter(tau_arr,ave_v2,c='green' ,s=2.5,label='36'+'nm')
    y_up_1 = savgol_filter(ave_v1+std_v1,11,5) # window length and polynomia order for the filter
    y_down_1 = savgol_filter(ave_v1-std_v1,11,5)
    y_up_2 = savgol_filter(ave_v2+std_v2,11,5)
    y_down_2 = savgol_filter(ave_v2-std_v2,11,5)
    plt.fill_between(tau_arr,y_up_1,y_down_1,alpha=0.1,color='blue')
    plt.fill_between(tau_arr,y_up_2,y_down_2,alpha=0.1,color='green')
    
    if estimation:  # Estimation is the estimated values of v_2 given v_1 (red squares)
        num = r_2/r_1
        plt.scatter(tau_arr*num,ave_v1/num,marker='s',s=4,facecolors='none',edgecolors='red',label='Estimation')
        
    plt.xlim(0,tau_arr[-1])
    plt.xlabel(r'$\tau \: (s)$')
    plt.ylabel(r'$<v> \: (\mu m/s)$')
    plt.legend()
    #plt.savefig('velocities.pdf')
    plt.show()

# Make sure that these files are in memory. Run find_velocity() first!
ave_v_12 = np.load('Data/ave_v_12.npz')['arr_0']*micrometer
std_v_12 = np.load('Data/std_v_12.npz')['arr_0']*micrometer
ave_v_36 = np.load('Data/ave_v_36.npz')['arr_0']*micrometer
std_v_36 = np.load('Data/std_v_36.npz')['arr_0']*micrometer

#plot_velocities(tau_arr,ave_v_12,std_v_12,ave_v_36,std_v_36,estimation=True) # <-- run this


# ------------ Determine tau_op ------------------ 
from scipy.optimize import curve_fit, fmin
from scipy.integrate import quad # quad integration

def diffusion_distr(x,t_end,b):
    gamma = 6*np.pi*eta*r
    diff_const = b*(thermal_energy*electron_charge)/gamma
    return 1/np.sqrt(4*diff_const*t_end)*np.exp(-x**2/(4*diff_const*t_end))

def g(tau,a,b):  # the analytic expression for <v>, only nearest-neighbour hopping
    I_1 = np.asarray([quad(diffusion_distr,L*alpha,L*(1+alpha),args=(3*t/4,b)) for t in tau])
    I_2 = np.asarray([quad(diffusion_distr,L*(-2+alpha),L*(-1+alpha),args=(3*t/4,b)) for t in tau])
    return (I_1[:,0]-I_2[:,0])*a/tau

def f(x,a,b): # turn the function around in order to use fmin()
    return -g(x,a,b)

def plot_fitted_velocity(tau_arr,ave_v1,ave_v2): # also plots curve-fitted curve
    global r
    ave_v1 = ave_v1
    ave_v2 = ave_v2
    plt.figure(figsize=(9,5))
    
    r = r_1
    popt_1, pcov = curve_fit(g,tau_arr,ave_v1)
    minimum_1 = fmin(f,0.5,args=(popt_1[0],popt_1[1],))
    plt.plot(tau_arr,g(tau_arr,popt_1[0],popt_1[1]),color='blue',linestyle='dashed',linewidth=1.6,label='12 nm curve fit')
    plt.scatter(tau_arr,ave_v1,s=4,c='blue',label='12 nm')
    plt.plot(tau_arr,g(tau_arr,L*micrometer,1),color='blue',linewidth=1.6,label='12 nm analytic')
    print('With r=',round(r*10**9,1),'\t tau_op=',minimum_1[0])
    print('Parameter a=',popt_1[0], 'b=',popt_1[1])

    r = r_2
    popt_2, pcov = curve_fit(g,tau_arr,ave_v2)
    minimum_2 = fmin(f,0.5,args=(popt_2[0],popt_2[1],))
    plt.plot(tau_arr,g(tau_arr,popt_2[0],popt_2[1]),color='green',linestyle='dashed',linewidth=1.6,label='36 nm curve fit')
    plt.scatter(tau_arr,ave_v2,s=4,c='green',label='36 nm')
    plt.plot(tau_arr,g(tau_arr,L*micrometer,1),color='green',linewidth=1.6,label='36 nm analytic')
    print('With r=',round(r*10**9,1),'\t tau_op=',minimum_2[0])
    print('Parameter a=',popt_2[0], 'b=',popt_2[1])
    
    plt.xlim(0,tau_arr[-1])
    plt.xlabel(r'$\tau \: (s)$')
    plt.ylabel(r'$<v> \: (\mu m/s)$')
    plt.legend()
    #plt.savefig('fitted_velocities.pdf')
    plt.show()
    
#plot_fitted_velocity(tau_arr,ave_v_12,ave_v_36) # <-- run this


# ---- Prints standard deviations at maximum velocity -----
tau_op_1 = 0.428
tau_op_2 = 1.253
#print('Velocity v_2 when tau=tau_op_1;',ave_v_36[np.abs(tau_op_1-tau_arr).argmin()])

#print('Standard deviation to v_max_12;', std_v_12[np.abs(tau_op_1-tau_arr).argmin()])
#print('Standard deviation to v_max_36;',std_v_36[np.abs(tau_op_2-tau_arr).argmin()])
#print('Standard deviation to tau;', tau_arr[1]-tau_arr[0])
