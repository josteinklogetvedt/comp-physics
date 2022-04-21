
from solver import *

# constants
tau = 1    # set to random value, just needed for parameter input
x_0 = np.asarray([0.0])  # initial condition

def plot_trajectory(x,t,dt,dU,savename=False):
    assert(time_criterion(dt,dU,r))
    y = x*micrometer
    plt.plot(t,y,color='black',linewidth=1.6)
    plt.ylabel(r'$x \: (\mu m)$')
    plt.xlabel(r'$t \: (s)$')
    plt.xlim(t[0]-dt,t[-1]+dt)
    plt.ylim(np.amin(y)-10,np.amax(y)+10)
    fac = round(dU/thermal_energy,1)
    plt.title(r'$\Delta U=$'+str(fac)+r'$k_b T$')
    if savename:
        plt.savefig(savename)
    plt.show()

def Boltzmann_distr(potential,dU,thermal_energy):
    return np.exp(-potential/thermal_energy)/(thermal_energy*(1-np.exp(-dU/thermal_energy)))

def plot_boltzdistr_of_x(x,dU,savename=False):
    assert(time_criterion(dt,dU,r))
    counts, bins = np.histogram(x*micrometer,bins=1000)
    
    x_axis = np.linspace(np.amin(x),np.amax(x),200)*micrometer
    tau_omega = 1 # random value
    u = U(x_axis/(micrometer*L),tau_omega*3/4,tau_omega)*dU # x needs to be dimensionless. Assert the potential is turned on.
    p = Boltzmann_distr(u,dU,thermal_energy)
    
    factor = np.mean(p)/np.mean(counts)    
    c,b = np.histogram(bins[:-1],bins,weights=counts*factor)   # each bin is given a weight
    scale = np.amax(p)
    c = c/scale                # rescale counts to be comparable to p
    p_squezeed = (p-np.amin(p))/(np.amax(p)-np.amin(p))  # gives a zoomed picture of p
    
    plt.scatter(b[:-1],c,s=1.2,c='blue',label='Computed')
    plt.plot(x_axis,p/scale,color='red',label='Boltzmann')
    plt.plot(x_axis,p_squezeed,color='red',linestyle='dashed')
    plt.xlabel(r'$x \: (\mu m)$')
    plt.ylabel(r'$p$')
    plt.ylim(0,np.amax(c)+0.1)
    plt.xlim(np.amin(b)-0.1,np.amax(b)+0.1)
    plt.legend()
    fac = round(dU/thermal_energy,1)
    plt.title(r'$\Delta U=$'+str(fac)+r'$k_b T$')
    if savename:
        plt.savefig(savename)
    plt.show()

def plot_boltzdistr_of_pot(pot,dU,savename=False):
    assert(time_criterion(dt,dU,r))
    counts, bins = np.histogram(pot/dU,bins=1000)
    
    u = np.linspace(np.amin(pot),np.amax(pot),200)
    p = Boltzmann_distr(u,dU,thermal_energy)
    
    factor = np.mean(p)/np.mean(counts)
    c,b = np.histogram(bins[:-1],bins,weights=counts*factor)  # scale counts to be comparable to p
    
    scale = np.amax(p)
    plt.scatter(b[:-1],c/scale,s=1.2,c='blue',label='Computed')
    plt.plot(u/dU,p/scale,color='red',label='Boltzmann')
    plt.xlim(0,1)
    plt.ylim(0,np.amax(c)/scale+0.05)
    plt.xlabel(r'$U/ \Delta U$')
    plt.ylabel(r'$p$')
    plt.legend()
    fac = round(dU/thermal_energy,1)
    plt.title(r'$\Delta U=$'+str(fac)+r'$k_b T$')
    if savename:
        plt.savefig(savename)
    plt.show()

N = 4*10**6    # number of steps (positions) we gather
dt = 3*10**-5   # time-step
t = np.asarray([i*dt for i in range(N)])

# -----with dU = 0.1*K_bT------
dU = 0.1*thermal_energy
x, pot = solver(t,x_0,tau,dU,r,flashing=False)    # r = r in constants.py
x = x[:,0]
pot = pot[:,0]

#plot_trajectory(x,t,dt,dU)   # <-- run these for plots
#plot_boltzdistr_of_x(x,dU)
#plot_boltzdistr_of_pot(pot,dU)

# ----with du=10*k_bT----
dU = 10*thermal_energy
x, pot = solver(t,x_0,tau,dU,r,flashing=False)
x = x[:,0]
pot = pot[:,0]

#plot_trajectory(x,t,dt,dU)  # <--- run these for plots
#plot_boltzdistr_of_x(x,dU)
#plot_boltzdistr_of_pot(pot,dU)

