from spin_chain import *

# Plots for task c), d), e) and f)

# Start with plots for c), d) and e):
def plot_spin_chain(f,alpha,savefig=False): # plot the chain of spins as arrows in the (x,z)-plane, for different times
    N = 9
    S_0 = np.zeros((N,3))
    S_0[:,2] = 1
    S_0[0,0] = 0.1    # tilt the first spin a bit
    S_0 = normalize(S_0)
    omega = 2*kappa*S_0[0,2]/beta
    t = np.linspace(0,omega,700)
    S = time_solver(S_0,t,f,alpha,J=1)

    chain = np.linspace(1,N,N)
    times = np.linspace(0,1,5)*t[-1]

    cmap = plt.cm.viridis
    cNorm  = colors.Normalize(vmin=0.0, vmax=t[-1]) # taken from https://stackoverflow.com/questions/18748328/plotting-arrows-with-different-color-in-matplotlib
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

    fig  = plt.figure()
    ax = plt.gca()
    scale = 0.5/S_0[0,0]

    for i,time in enumerate(times):
        idx = np.abs(t-time).argmin()
        spins = S[idx,:,:]
        for j in range(N):
            colorVal = scalarMap.to_rgba(time)
            r = np.sqrt(spins[j,0]**2+spins[j,1]**2)*np.sign(spins[j,0])

            ax.arrow(chain[j],0,r*scale,spins[j,2],color=colorVal,width=0.03,shape='full',head_width=0.06,length_includes_head=True)
    cbar = fig.colorbar(scalarMap,ax=ax,orientation='horizontal',pad=0.02)
    cbar.set_label(r'$\tilde{t}$',rotation=0)
    plt.xticks([])
    plt.ylabel(r'$S_z$')
    plt.ylim(0,1.3)
    plt.xlim(0.2,N+0.8)
    if savefig:
        plt.savefig(savefig)
    plt.show()
    
#plot_spin_chain(spin_chain_no_PBC,alpha=0.1)  # task c)

def plot_spin_chain_S_xy(f,alpha,savefig=False): # Plot the chain of spins as arrows in the (x,y)-plane, for different times
    N = 9
    S_0 = np.zeros((N,3))
    S_0[:,2] = 1
    S_0[0,0] = 0.1    
    S_0 = normalize(S_0)
    omega = 2*kappa*S_0[0,2]/beta
    t = np.linspace(0,omega,700)
    S = time_solver(S_0,t,f,alpha,J=1)

    chain = np.linspace(1,N,N,dtype=int)*2.0
    times = np.linspace(0,1,5)*t[-1]

    cmap = plt.cm.viridis
    cNorm  = colors.Normalize(vmin=0.0, vmax=t[-1]) # taken from https://stackoverflow.com/questions/18748328/plotting-arrows-with-different-color-in-matplotlib
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

    fig  = plt.figure(figsize=(7.4,4.8))
    ax = plt.gca()
    jumps = 35
    for i in range(len(chain)):

        spins = S[::jumps,i,:]
        for j,time in enumerate(t[::jumps]):
            colorVal = scalarMap.to_rgba(time)
            scale = np.sqrt(spins[j,0]**2 + spins[j,1]**2)
            if not (np.isclose(spins[j,0],0.0) or np.isclose(spins[j,0],0,0)):
                ax.arrow(chain[i],0,spins[j,0]/scale,spins[j,1]/scale,color=colorVal,width=0.02,shape='full',head_width=0.06,length_includes_head=True)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$S_x$')
    plt.ylabel(r'$S_y$')
    cbar = fig.colorbar(scalarMap,ax=ax,orientation='horizontal',pad=0.08)
    cbar.set_label(r'$\tilde{t}$',rotation=0)

    if savefig:
        plt.savefig(savefig)
    plt.show()

#plot_spin_chain_S_xy(spin_chain_no_PBC,alpha=0.1)  # c)
#plot_spin_chain_S_xy(spin_chain_w_PBC,alpha=0.0)   # e)


def plot_Sx_and_Sy(f,alpha,savefig=False):  # Plot S_x and S_y as a function of time for different lattice sites
    N = 9
    S_0 = np.zeros((N,3))
    S_0[:,2] = 1
    S_0[0,0] = 0.1 # tilt the first spin a bit 
    S_0 = normalize(S_0) 
    omega = 2*kappa*S_0[0,2]/beta
    t = np.linspace(0,omega*2.5,700,endpoint=False)
    S = time_solver(S_0,t,f,alpha,J=1)
    
    points = [0,1,2]
    if alpha == 0.0:
        points = [0,1,8]   

    colors = ['blue','green','black']

    fig, axs = plt.subplots(ncols=1,nrows=2,sharex=True)
    for k, ax in enumerate(axs):
        for i in range(len(points)):
            ax.plot(t,S[:,points[i],k],color=colors[i],label='Site ' + r'$i=$'+str(points[i]))
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlabel(r'$\tilde{t}$')
    axs[0].set_ylabel(r'$S_x$')
    axs[1].set_xlabel(r'$\tilde{t}$')
    axs[1].set_ylabel(r'$S_y$')

    if savefig:
        plt.savefig(savefig)
    plt.show()

#plot_Sx_and_Sy(spin_chain_no_PBC,alpha=0.1) # c)
#plot_Sx_and_Sy(spin_chain_no_PBC,alpha=0.0) # d)
#plot_Sx_and_Sy(spin_chain_w_PBC,alpha=0.0)  # e)

def plot_wave(alpha,savefile=False):  # Plot S_x along the lattice sites for different times
    N = 20
    T_end = 4
    dt = 0.01
    S_0 = np.zeros((N,3))
    S_0[:,2] = 1
    S_0[0,0] = 0.1   # tilt the first spin a bit
    num = int(T_end/dt)
    t = np.linspace(0,T_end,num)
    S = time_solver(S_0,t,spin_chain_no_PBC,alpha,J=1)

    times = [1.0,1.5,2.0,3.5]

    cmap = plt.cm.seismic
    cNorm  = colors.Normalize(vmin=times[0], vmax=t[-1]) 
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

    S_x = S[:,:,0]
    x = np.linspace(0,N-1,200)

    for i in range(len(times)):
        idx = np.abs(times[i]-t).argmin()
        f = interp1d(np.arange(N),S_x[idx,:],kind='cubic')   # Interpolation for smooth curve
        colorVal = scalarMap.to_rgba(t[idx])
        
        plt.plot(x,f(x),label=r'$\tilde{t}=\:$' + str(round(t[idx],1)),color=colorVal)
    plt.xlabel('Lattice site')
    plt.ylabel(r'$S_x$')
    plt.legend()
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
#plot_wave(alpha=0.1) # c)
#plot_wave(alpha=0.0) # d)


# ------ Looking for the ground state -------
# Task f) # use same approximations as above, use PBC and N = even 
# Randomly initialize spins, run for a longer time to find equilibrium

def plot_magnetization(savefile=False): # plots magnetization for both the FM and AFM case
    N = 20
    alpha = 0.1
    S_0 = np.random.uniform(low=-1,high=1,size=(N,3))     # randomly initialize spin
    dt = 0.02
    T_end = 50
    num = int(T_end/dt)
    t = np.linspace(0,T_end,num)

    J = [-1.0,1.0]
    colors = ['dodgerblue','blue']
    symbols = [r'$<$',r'$>$']
    for i in range(2):
        S = time_solver(S_0,t,spin_chain_w_PBC,alpha,J=J[i])
        S_z = S[:,:,2]
        M = (1/N)*np.sum(S_z,axis=1)
        plt.plot(t,M,color=colors[i],label=r'$J$'+symbols[i]+'0')
    plt.legend(loc=7)
    plt.xlabel(r'$\tilde{t}$')
    plt.ylabel(r'$M$')
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
#plot_magnetization()

def plot_GS(J,savefile=False): # Plots S_y and S_z for all sites at different times
    N = 20
    alpha = 0.1
    S_0 = np.random.uniform(low=-1,high=1,size=(N,3))     # randomly initialize spin
    dt = 0.02
    T_end = 50
    num = int(T_end/dt)
    t = np.linspace(0,T_end,num)
    
    S = time_solver(S_0,t,spin_chain_w_PBC,alpha,J)
    
    if J<0: 
        T_end = T_end/4   # the AFM reaches equilibrium much faster
        
    times = [T_end//4,T_end//2,3*T_end//4,T_end]
    components = [r'$S_y$',r'$S_z$']
    mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
    
    fig, axs = plt.subplots(nrows=2,sharex=True)
    for i, ax in enumerate(axs):
        for j in range(len(times)):
            idx = np.abs(t-times[j]).argmin()
            ax.scatter(np.arange(N),S[idx,:,i+1],label=r'$\tilde{t}=$'+str(round(t[idx],1)),s=12)
            ax.set_ylabel(components[i])
            ax.set_xlim(-0.5,N-0.5)
    
    axs[0].legend(loc='upper center',bbox_to_anchor=(0.5,1.25),ncol=len(times))
    axs[1].set_xlabel('Lattice site')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile)
    plt.show()

#plot_GS(1.0)  # FM
#plot_GS(-1.0)  # AFM