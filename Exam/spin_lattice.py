from utils import *

# Code for solutions to task g), h) and i)

# Constants and parameters used in these tasks
gamma = 1.76*10**(11)
mu = 9.274*10**(-24)
J = 1*10**(-3)*1.6*10**(-19)
k_b = 1.3807*10**(-23)
ps = 1*10**(-12)
d_z = 0.1*J
alpha = 0.9


@nb.njit
def sum_NN_lattice(S):  # S is an N*N x 3 matrix. Performs N.N. summation as if the spins are on a NxN lattice. Use PBC.
    N_squared = S.shape[0]
    N = int(np.sqrt(N_squared))
    q = lambda j,i : j*N+i
    res = np.zeros(S.shape)
    
    for i in range(1,N-1):
        res[q(0,i)] = S[q(0,i-1)] + S[q(0,i+1)] + S[q(1,i)] + S[q(N-1,i)]
        res[q(N-1,i)] = S[q(N-1,i-1)] + S[q(N-1,i+1)] + S[q(0,i)] + S[q(N-2,i)]
        res[q(i,0)] = S[q(i,N-1)] + S[q(i,1)] + S[q(i-1,0)] + S[q(i+1,0)]
        res[q(i,N-1)] = S[q(i,N-2)] + S[q(i,0)] + S[q(i-1,N-1)] +S[q(i+1,N-1)]
        for j in range(1,N-1):
            res[q(j,i)] = S[q(j-1,i)] + S[q(j+1,i)] + S[q(j,i-1)] + S[q(j,i+1)]
            
    res[0] = S[1] + S[N-1] + S[N] + S[N*(N-1)]        
    res[N-1] = S[0] + S[N-2] + S[N_squared-1] + S[2*N-1]
    res[N*(N-1)] = S[N_squared-1] + S[q(N-1,1)] + S[0] + S[q(N-2,0)]
    res[N_squared-1] = S[q(N-1,N-2)] + S[N*(N-1)] + S[q(N-2,N-1)] + S[N-1]

    return res

@nb.njit
def spin_lattice_PBC(S,B,noise): # derivative function d_t S = f(S)
    
    F = (2*J/mu)*sum_NN_lattice(S) + noise
    F[:,2] += B + (2*d_z/mu)*S[:,2] # magnetic field and d_z term
    S_cross_F = calc_cros(S,F)
    
    return (-gamma)*(1/(1+alpha**2))*(S_cross_F + alpha* calc_cros(S,S_cross_F))

@nb.njit
def time_solver(S_0,t,B,Th): # Th is the relative thermal energy, B is the magnetic field
    S = np.zeros((len(t),*S_0.shape),dtype=S_0.dtype)
    S[0] = normalize(S_0)
    dt = t[1]-t[0]
    prefac = np.sqrt(2*alpha*J*Th/(gamma*mu*dt))

    for i in range(1,len(t)):
        noise = np.random.normal(loc=0.0,scale=1.0,size=S_0.shape)*prefac
        S[i] = normalize(heun_step(S[i-1],dt,spin_lattice_PBC,B,noise))

    return S

def phase_diagram(T_end,t_eq,N,B,Th_arr,savefile=False): # Performs temporal average from t_eq to T_end for each Th in Th_arr
    S_0 = np.zeros((N*N,3))
    S_0[:,2] = np.ones(N*N)
    
    dt = 0.001*ps
    num = int(T_end/dt)
    t = np.linspace(0,T_end,num)
    idx = np.abs(t-t_eq).argmin()

    M = np.zeros_like(Th_arr)
    M_std = np.zeros_like(Th_arr)
    for i,Th in enumerate(Th_arr):
        S = time_solver(S_0,t,B,Th)
        m = (1/N**2)*np.sum(S[:,:,2],axis=1)
        M[i] = np.average(m[idx:])
        M_std[i] = np.std(m[idx:])
        
    if savefile:
        np.savez(savefile,M,M_std)  # save to file
    return M, M_std