import numpy as np
import matplotlib.pyplot as plt

h_bar = 6.5821*10**(-16)  # eV*s
alpha = 1/(137.035)
E_e = 0.511*10**(6) # eV
c = 3*10**8

np.random.seed(42)

def Klein_Nishina(E_photon,theta):
    k = E_photon/E_e
    E_photon_ratio = 1/(1+k*(1-np.cos(theta)))
    prefac = (1/2)*(h_bar*alpha*c/E_e)**2
    return prefac*E_photon_ratio**2*(E_photon_ratio + E_photon_ratio**(-1) - np.sin(theta)**2)

def plot_angle_distr(savefile=False):
    E_photons = np.linspace(0.1,1.5,5)*10**6
    theta = np.linspace(0,2*np.pi,100)
    
    for i in range(len(E_photons)):
    
        prob_distr = Klein_Nishina(E_photons[i],theta)
        plt.plot(theta*180/np.pi,prob_distr,label=r'$E_{\gamma}$='+str(round(E_photons[i]*10**(-6),1))+' MeV')

    plt.ylabel(r'$\frac{d\sigma}{d\Omega} \: (m^2 sr^{-1})$')
    plt.xlabel(r'$\theta$ (degree)')
    plt.legend()
    plt.xlim(0,360)
    if savefile:
        plt.savefig(savefile)
    plt.show()

def plot_angle_distr_polar(init_angle, savefile=False):
    E_photons = np.linspace(0.1,1.5,5)*10**6
    theta = np.linspace(0,2*np.pi,100)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for i in range(len(E_photons)):
    
        prob_distr = Klein_Nishina(E_photons[i],theta)
        ax.plot(theta+init_angle,prob_distr)
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
#plot_angle_distr()
#plot_angle_distr_polar(0)

from scipy import interpolate

def sample(g,*args):
    x = np.linspace(0,2*np.pi,10**5)
    y = g(*args,x)                        # probability density function, pdf
    cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(cdf_y,x,fill_value='extrapolate',assume_sorted=True)    # this is a function
    return inverse_cdf

def plot_prob_distr_random(savefile=False):
    E_photon = 0.1*10**6
    points = 1*10**5

    theta_axis = np.linspace(0,2*np.pi,100)
    p_KL = Klein_Nishina(E_photon,theta_axis)

    uniform = np.random.rand(points)
    CDF = sample(Klein_Nishina,E_photon)
    thetas = CDF(uniform)

    counts, bins = np.histogram(thetas*180/np.pi,bins=80)
    factor = np.mean(p_KL)/np.mean(counts)
    c, b = np.histogram(bins[:-1],bins,weights=counts*factor)
    plt.scatter(b[:-1],c,s=3.5,c='blue',label='Computed')

    plt.plot(theta_axis*180/np.pi,p_KL,linestyle='dashed',color='red',label='Klein-Nishina')
    plt.xlim(0-5,360+5)
    plt.ylabel(r'$\frac{d\sigma}{d\Omega} \: (m^2 sr^{-1})$')
    plt.xlabel(r'$\theta$ (degree)')
    plt.legend(loc='upper center')
    plt.title(r'$E_{\gamma}=$'+str(round(E_photon*10**(-6),1))+' MeV')
    if savefile:
        plt.savefig(savefile)
    plt.show()
    
#plot_prob_distr_random() 

