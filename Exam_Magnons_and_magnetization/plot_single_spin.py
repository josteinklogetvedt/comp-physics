from single_spin import *


# Task a)
# Have alpha = 0. Plot analytical to compare
def plot_single_spin(savefig=False):
    alpha = 0.0
    S_0 = normalize_1D(np.asarray([1,5,25],dtype=np.float64))
    omega = 2*kappa*S_0[2]/beta
    t = np.linspace(0,5*omega,100)
    S = time_solver(S_0,t,alpha)

    t_refined = np.linspace(0,t[-1],200)
    S_x_an, S_y_an = analytical(S_0,omega,t_refined)
    plt.plot(t_refined/omega,S_x_an,color='red',linestyle='dashed')
    plt.plot(t_refined/omega,S_y_an,color='blue',linestyle='dashed')

    plt.plot(t/omega,S[:,0],color='red',label=r'$S_x$')
    plt.plot(t/omega,S[:,1],color='blue',label=r'$S_y$')
    plt.plot(t/omega,S[:,2],color='green',label=r'$S_z$')
    plt.legend()
    plt.xlabel(r'$\tilde{t}/\omega$')
    plt.ylabel(r'$S$')
    print(f'The initial spin is {S_0}, and the final spin is {S[-1,:]}.')
    if savefig:
        plt.savefig(savefig)
    plt.show()

#plot_single_spin()


# ---- Task b) -----
alpha = 0.05
def plot_single_spin_damping(savefig=False):
    S_0 = normalize_1D(np.asarray([1,5,25],dtype=np.float64))
    omega = 2*kappa*S_0[2]/beta
    tau = 1/(alpha*omega)
    t = np.linspace(0,3*tau,300)
    S = time_solver(S_0,t,alpha)

    plt.plot(t/omega,S[:,0],color='red',label=r'$S_x$')
    plt.plot(t/omega,S[:,1],color='blue',label=r'$S_y$')
    plt.plot(t/omega,S[:,2],color='green',label=r'$S_z$')
    plt.legend()
    plt.xlabel(r'$\tilde{t}/\omega$')
    plt.ylabel(r'$S$')
    print(f'The initial spin is {S_0}, and the final spin is {S[-1,:]}.')
    if savefig:
        plt.savefig(savefig)
    plt.show()

#plot_single_spin_damping()


def f(x,a):  # trial-function used for curve-fitting. Find the best value of tau fitted to our data.
    global S_first_maxima
    return S_first_maxima*np.exp(-x/a)

# Find tau and compare with tau=1/(alpha*omega)
def plot_damping(savefig=False):
    S_0 = normalize_1D(np.asarray([1,5,25],dtype=np.float64))
    omega = 2*kappa*S_0[2]/beta
    tau = 1/(alpha*omega)
    N = 12
    num = 300
    t = np.linspace(0,N*omega,num)
    S = time_solver(S_0,t,alpha)
    S_y = S[:,1]
 
    indices = argrelextrema(S_y,np.greater)[0] # Found from  https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array

    global S_first_maxima  # used in curve_fit function, will not parametarize this variable
    S_first_maxima = S_y[indices[0]]
    
    popt, pcov = curve_fit(f,t[indices],S_y[indices]) # scipy's curve_fit

    plt.plot(t,S_y[indices[0]]*np.exp(-t/tau),color='black',linestyle='dashed',label='Analytical '+ r'$\tau=\:$'+str(round(tau,2)))
    plt.scatter(t[indices],S_y[indices],color='blue')
    plt.plot(t,f(t,popt[0]),color='black',label='Curve_fit ' + r'$\tau=\:$'+str(round(popt[0],2)))
    plt.plot(t,S_y,color='blue',label=r'$S_y$')
    plt.legend()
    plt.xlabel(r'$\tilde{t}$')
    plt.ylabel(r'$S$')
    print(f'The initial spin is {S_0}, and the final spin is {S[-1,:]}.')
    if savefig:
        plt.savefig(savefig)
    plt.show()
    
#plot_damping()