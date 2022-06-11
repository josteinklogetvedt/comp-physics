
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

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

def sigma_PE(Z,E_photon):
    return 3*10**(12)*Z**4*(E_photon)**(-3.5)

class Length:
    def __init__(self,Z):
        Z_low = 3 # Lithium
        Z_high = 82 # Lead
        density_litium = 0.534
        density_lead = 11.29
        self.lengths = np.zeros(3)
        if Z == Z_low:
            mu = np.asarray([1.289*10**(-1),7.532*10**(-2),6.698*10**(-2)])*density_litium*10**(2) # conversion to meter
            self.lengths = 1/mu
        elif Z == Z_high:
            mu = np.asarray([5.549,1.614*10**(-1),1.248*10**(-1)])*density_lead*10**(2)
            self.lengths = 1/mu
        else:
            print('Error; wrong Z number')
        #print((self.lengths[0]*0.05+self.lengths[1]*0.8+self.lengths[2]*0.15))  # mean length in 1 dimension 
            
class Energy:
    def __init__(self,num_photons):
        self.initial_energies = np.asarray([0.135,0.525,0.615])*10**6
        self.initial_counts =  np.asarray([850,13600,2550])
        self.num_photons = num_photons
        self.random_energies = np.random.choice(self.initial_energies,size=self.num_photons,p = self.initial_counts/np.sum(self.initial_counts))
    def get_E_init(self):
        return self.random_energies
    
class Structure(Energy,Length):
    def __init__(self,Z,num_photons):
        Energy.__init__(self,num_photons)
        Length.__init__(self,Z)
        self.pdfs = []
        self.l_0 = np.zeros(num_photons)
        self.energy_array = np.concatenate((np.asarray([0.001,0.005])*10**6,np.linspace(0.01,0.615,50)*10**6))  # own made-up array

        for i in range(len(self.initial_energies)):
            idx = np.where(self.random_energies == self.initial_energies[i])[0]
            self.l_0[idx] = self.lengths[i]
            
        for i in range(len(self.energy_array)):
            self.pdfs.append(self.inverse_cdf(Klein_Nishina,self.energy_array[i]))
          
    def inverse_cdf(self,g,E):
        x = np.linspace(-np.pi,np.pi,10**5)
        y = g(E,x)                     
        cdf_y = np.cumsum(y)          
        cdf_y = cdf_y/cdf_y.max()      
        inverse_cdf = interpolate.interp1d(cdf_y,x,fill_value='extrapolate',assume_sorted=True)    # this is a function
        return inverse_cdf
    
    def get_l_0(self):
        return self.l_0
    
    def __call__(self,E):
        uniform = np.random.rand(len(E))
        thetas = np.zeros(len(E))
        for i in range(len(E)):
            idx = np.abs(E[i]-self.energy_array).argmin()
            thetas[i] = self.pdfs[idx](uniform[i])
        return thetas

def MC_photon_transport_multiple(Z,num_photons):
    Z = np.float64(Z)
    
    Prob_structure = Structure(Z,num_photons)
    l_0 = Prob_structure.get_l_0()
    E = Prob_structure.get_E_init()
    E_new = np.copy(E)

    incoming_angles = np.zeros(num_photons)
    E_lost = []
    mean_pos = [[0.0,0.0]]
    x_last = np.zeros(len(E))
    y_last = np.zeros(len(E))
    
    not_absorbed=True
    index_arr = np.ones(len(E),dtype=bool)

    while not_absorbed:

        photons_left = len(E[index_arr])
        random_numbers = np.random.rand(photons_left)
        index_arr[index_arr] = np.where(random_numbers <= (1-np.exp(-sigma_PE(Z,E[index_arr])*l_0[index_arr])),np.zeros(photons_left),np.ones(photons_left))

        if np.sum(index_arr) == 0:
            not_absorbed = False
        else:
            k = E[index_arr]/E_e
            theta_defl = Prob_structure(E[index_arr])
            incoming_angles[index_arr] += theta_defl

            E_new[index_arr] = E[index_arr]/(1+k*(1-np.cos(theta_defl)))
            E_lost.append(np.mean(E[index_arr]-E_new[index_arr]))  
            E[index_arr] = E_new[index_arr]
            
            x_last[index_arr] += l_0[index_arr]*np.cos(incoming_angles[index_arr])
            y_last[index_arr] += l_0[index_arr]*np.sin(incoming_angles[index_arr])
            mean_pos.append([np.mean(x_last),np.mean(y_last)])
            
    m_pos = np.asarray(mean_pos)
    return x_last,y_last,np.asarray(E_lost),m_pos[:-1,:]    # drop the last element where no energy is lost due to PE

def MC_photon_transport_modified(Z,num_photons):
    Z = np.float64(Z)

    Prob_structure = Structure(Z,num_photons)
    l_0 = Prob_structure.get_l_0()
    l = np.copy(l_0)
    E = Prob_structure.get_E_init()
    E_new = np.copy(E)

    incoming_angles = np.random.uniform(0,2*np.pi,size=num_photons)
    E_lost = []
    mean_pos = [[0.0,0.0]]
    x_last = np.zeros(len(E))
    y_last = np.zeros(len(E))
    
    not_absorbed=True
    index_arr = np.ones(len(E),dtype=bool)
    
    while not_absorbed:

        photons_left = len(E[index_arr])
        eta = np.random.rand(photons_left)
        l[index_arr] = l_0[index_arr]*(1-eta)
        random_numbers = np.random.rand(photons_left)
        index_arr[index_arr] = np.where(random_numbers <= (1-np.exp(-sigma_PE(Z,E[index_arr])*l[index_arr])),np.zeros(photons_left),np.ones(photons_left))

        if np.sum(index_arr) == 0:
            not_absorbed = False
        else:
            k = E[index_arr]/E_e
            theta_defl = Prob_structure(E[index_arr])
            incoming_angles[index_arr] += theta_defl
            
            E_new[index_arr] = E[index_arr]/(1+k*(1-np.cos(theta_defl)))
            E_lost.append(np.mean(E[index_arr]-E_new[index_arr]))
            E[index_arr] = E_new[index_arr]
            
            x_last[index_arr] += l[index_arr]*np.cos(incoming_angles[index_arr])
            y_last[index_arr] += l[index_arr]*np.sin(incoming_angles[index_arr])
            mean_pos.append([np.mean(x_last[index_arr]),np.mean(y_last[index_arr])])
    
    m_pos = np.asarray(mean_pos)       
    return x_last,y_last,np.asarray(E_lost),m_pos[:-1,:]

def plot_energy(e_lost,Z,savefile=False):
    plt.plot(np.arange(1,len(e_lost)+1),e_lost,color='blue',marker='o')
    plt.title(r'$Z=$'+str(Z))
    plt.xlabel('Iterations')
    plt.yscale('log')
    plt.ylabel('eV')
    if savefile:
        plt.savefig(savefile)
    plt.show()

def plot_last_pos(x_last,y_last,Z,savefile=False):
    print('Mean y;',np.mean(y_last))
    print('Mean x;',np.mean(x_last))
    r = np.sqrt(x_last**2+y_last**2)
    print(f'Mean r: {np.mean(r)} with standard deviation: {np.std(r)}')
    
    conversion = 10**3
    conversion_string = 'mm'
    if Z == 3:
        conversion = 1
        conversion_string = 'm'
    x_last *= conversion
    y_last *= conversion
    plt.scatter(x_last,y_last,s=0.5,color='blue')
    plt.scatter(np.mean(x_last),np.mean(y_last),c='red')
    plt.xlabel(r'$x$'+' ('+conversion_string+')')
    plt.ylabel(r'$y$'+' ('+conversion_string+')')
    plt.title(r'$Z=$'+str(Z))
    if savefile:
        plt.savefig(savefile)
    plt.show()

import matplotlib as ml
def plot_trajectory_with_energy(pos,e_lost,Z,savefile=False):
    conversion = 10**3 # (mm)
    conversion_string = 'mm'
    if Z == 3:
        conversion = 1
        conversion_string = 'm'
    pos = pos*conversion
    
    plt.plot(pos[:,0],pos[:,1],linewidth=1.5,color='gray',linestyle='dashed',alpha=0.5)
    sc = plt.scatter(pos[:,0],pos[:,1],norm=ml.colors.LogNorm(vmin=np.amin(e_lost),vmax=np.amax(e_lost)),c=e_lost)
    cbar = plt.colorbar(sc)
    cbar.set_label('eV',rotation=0)
    ax = plt.gca()
    ax.set_ylabel(r'$y$'+' ('+conversion_string+')')
    ax.set_xlabel(r'$x$'+' ('+conversion_string+')')

    x_max = np.amax(pos[:,0])
    x_min = np.amin(pos[:,0])
    y_max = np.amax(pos[:,1])
    y_min = np.amin(pos[:,1])
    delta_x = (x_max-x_min)/10
    delta_y = (y_max-y_min)/10
    plt.plot(np.zeros(20),np.linspace(y_min-delta_y,y_max+delta_y,20),color='gray',linewidth=1.5)
    plt.plot(np.linspace(x_min-delta_x,x_max+delta_x,20),np.zeros(20),color='gray',linewidth=1.5)
    plt.xlim(x_min-delta_x,x_max+delta_x)
    plt.ylim(y_min-delta_y,y_max+delta_y)
    plt.grid()
    plt.title('Z='+str(Z))
    if savefile:
        plt.savefig(savefile)
    plt.show()

#### --- Run simulations ----
N = 1*10**5

# With Lithium
Z_low = 3
#x,y,e_lost,mean_pos = MC_photon_transport_multiple(Z_low,N)
#plot_last_pos(x,y,Z_low)
#plot_energy(e_lost,Z_low)
#x,y,e_lost,mean_pos = MC_photon_transport_multiple(Z_low,1)
#plot_trajectory_with_energy(mean_pos,e_lost,Z_low)

# With Lead
Z_high = 82
#x,y,e_lost,mean_pos = MC_photon_transport_multiple(Z_high,N)
#plot_last_pos(x,y,Z_high)
#plot_energy(e_lost,Z_high)
#x,y,e_lost,mean_pos = MC_photon_transport_multiple(Z_high,1)
#plot_trajectory_with_energy(mean_pos,e_lost,Z_high)


# ------ Modified method ---------
# Lithium
#x,y,e_lost,mean_pos = MC_photon_transport_modified(Z_low,N)
#plot_last_pos(x,y,Z_low)
#plot_energy(e_lost,Z_low)
#x,y,e_lost,mean_pos = MC_photon_transport_modified(Z_low,1)
#plot_trajectory_with_energy(mean_pos,e_lost,Z_low)

# Lead
#x,y,e_lost,mean_pos = MC_photon_transport_modified(Z_high,N)
#plot_last_pos(x,y,Z_high)
#plot_energy(e_lost,Z_high)
#x,y,e_lost,mean_pos = MC_photon_transport_modified(Z_high,1)
#plot_trajectory_with_energy(mean_pos,e_lost,Z_high)