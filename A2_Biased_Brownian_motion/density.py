

from solver import *

# constants 
dU = 80  #eV
dt = 2*10**-5
r_1 = r
r_2 = 3*r

def diffusion_solution(x,t_end,r,N): # analtic expression for density
    gamma = 6*np.pi*eta*r
    diff_const = (thermal_energy*electron_charge)/gamma
    return N/np.sqrt(4*diff_const*t_end)*np.exp(-x**2/(4*diff_const*t_end))

@nb.njit
def run_simulation(T_end,number_of_particles,r_arr,tau_arr,step=Euler_step): #  wrapper around solver function. Can send in r- and tau-arrays
    t = np.arange(0,T_end+dt,dt)
    x = np.zeros((len(r_arr),number_of_particles))
    x_0 = np.zeros(number_of_particles)

    for i in range(len(r_arr)):
        x[i,:] = solver_only_last(t,x_0,tau_arr[i],dU,r_arr[i],step=step)

    return x, t

def plot_density_without_potential(T_end,number_of_particles,r_arr,colors,plot_n=True): 

    X,t = run_simulation(T_end,number_of_particles,r_arr,np.zeros(len(r_arr)),step=Brownian_step)
    num_of_points = 80
    
    assert(num_of_points<number_of_particles)

    for i in range(len(r_arr)):
        assert(time_criterion(dt,dU,r_arr[i]))
        x = X[i,:]*micrometer
        counts, bins = np.histogram(x,bins=num_of_points)
        r = r_arr[i]*nanometer
        
        axis = np.linspace(np.amin(x),np.amax(x),100)
        n = diffusion_solution(axis/micrometer,t[-1],r_arr[i],number_of_particles)
        n = n/np.amax(n)
        factor = np.mean(n)/np.mean(counts)
        c,b = np.histogram(bins[:-1],bins,weights=counts*factor)  # scale counts
        plt.scatter(b[:-1],c,s=2.5,c=colors[i],marker='x',label=str(round(r,1))+' nm') 

        if plot_n:  # plot the analytic density function
            plt.plot(axis,n,color=colors[i],label='Analytic',linewidth=1.6)

    plt.title('T='+str(T_end)+' (s)')
    plt.xlabel(r'$x \: (\mu m)$')
    plt.ylabel(r'$n(x,T)$')
    plt.legend()    
    plt.show()


number_of_particles = 1000  # for each type r_1 and r_2
T_end = 10
r_arr = np.asarray([r_1,r_2])
colors = ['red','blue']
#plot_density_without_potential(T_end,number_of_particles,r_arr,colors)  # <- run this. NB; have not added this in the report


def plot_density(f,cycles,r,number_of_particles): # cycles is an array. Plots density as a function of time, with flashing. 
    tau = 1/f
    fig, axs = plt.subplots(len(cycles),figsize=(9,5))
    assert(time_criterion(dt,dU,r))
    
    for i in range(len(cycles)):
        T_end = tau*cycles[i]      
        x, t = run_simulation(T_end,number_of_particles,np.asarray([r]),np.asarray([tau]))
        x = x[0,:]
        b = 80 # number of bins
        height = 20 # just some value
        counts, bins = np.histogram(x,bins=b)
        if i ==0:
            axis = bins[:-1]*100
            levels = 3
        else:
            axis = (bins[:-1] + (bins[1]-bins[0])/2)*micrometer
            levels = 5
            
        image = np.outer(np.ones(height),counts)
        xv, yv = np.meshgrid(axis,np.linspace(0,1,height),indexing='xy')
        axs[i].contourf(xv,yv,image,cmap='gray',levels=levels)
        axs[i].get_yaxis().set_visible(False)
        proxy1 = plt.Rectangle((0, 0), 0.2, 0.2, fc='none', ec='none',alpha=0.3, linewidth=3, label=str(cycles[i])+' cycles') # add label
        axs[i].patches += [proxy1]
        axs[i].legend()
        
    axs[-1].set_xlabel(r'$x \: (\mu m)$')
    #plt.savefig('density_wflash.pdf')
    plt.show()

freq = 0.7
cycles = np.asarray([0,10,20])
number_of_particles = 2000  
#plot_density(freq,cycles,r_1,number_of_particles)  # <-- run this

def plot_density_both_particles(tau_op,cycle,r_arr,number_of_particles): # cycle; float. Plot density asaf of time, for both particles.

    assert(time_criterion(dt,dU,r_arr[0]))
    assert(time_criterion(dt,dU,r_arr[1]))
    
    T_end = tau_op*cycle 
    X, t = run_simulation(T_end,number_of_particles,r_arr,np.asarray([tau_op,tau_op]))

    X = X*micrometer
    l = L*micrometer
    x_max = np.amax(X.flatten())
    x_min = np.amin(X.flatten())
    bin_min = x_min//l
    if x_min % l < alpha*l:
        bin_min -= 1
    bin_max = x_max//l 
    if x_max % l > alpha*l:
        bin_max += 1
    bin_edges = np.asarray([bin_min*l-(1-alpha)*l])
    for i in range(int(bin_min),int(bin_max)+1):
        bin_edges = np.append(bin_edges,i*l+alpha*l)
        
    print('Mean and std for particle 1;', np.mean(X[0,:]),np.std(X[0,:]))
    print('Mean and std for particle 2;', np.mean(X[1,:]),np.std(X[1,:]))
    print('Resolution;',np.abs(np.mean(X[0,:])-np.mean(X[1,:]))/(0.5*np.abs(np.std(X[0,:])-np.std(X[1,:]))) )
     
    counts_1,bins_1 = np.histogram(X[0,:],bins=bin_edges)
    counts_2,bins_2 = np.histogram(X[1,:],bins=bin_edges)
    counts_1 = counts_1/number_of_particles
    counts_2 = counts_2/number_of_particles
    n = 30
    width = l/30
    for i in range(len(counts_1)):
        if i == 0:
            plt.plot(np.ones(n)*(bin_edges[i+1]-alpha*l + width),np.linspace(0,counts_1[i],n),color='blue',linewidth=3,label='12 nm')
            plt.plot(np.ones(n)*(bin_edges[i+1]-alpha*l - width),np.linspace(0,counts_2[i],n),color='green',linewidth=3,label='36 nm')
        else:
            plt.plot(np.ones(n)*(bin_edges[i+1]-alpha*l + width),np.linspace(0,counts_1[i],n),color='blue',linewidth=3)
            plt.plot(np.ones(n)*(bin_edges[i+1]-alpha*l - width),np.linspace(0,counts_2[i],n),color='green',linewidth=3)
            
    plt.ylim(0,np.amax(counts_2))
    plt.xlim(x_min-10,x_max+10)
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.xlabel(r'$x \: (\mu m)$')
    plt.ylabel(r'$n(x,T)$')
    plt.legend()
    #plt.savefig('density_two_particles_wflash.pdf')
    plt.show()
    
tau_op = 0.428    # the obtained value in velocity.py
cycle = 150
r_arr = np.asarray([r_1,r_2])
number_of_particles = 1000  # for each particle type
#plot_density_both_particles(tau_op,cycle,r_arr,number_of_particles)  # <-- run this