import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Gaussian_distr():
    return np.random.standard_normal(1)[0]

def find_time_intervals(N):
    x = Gaussian_distr()
    intervals = np.zeros(N)
    n = 1
    i = 0
    
    #x_total = [x]
    while 0 in intervals:
        
        if x >= 0:
            while x >= 0:
                x += Gaussian_distr()
                #x_total.append(x)
                n += 1
        else:
            while x <= 0:
                x += Gaussian_distr()
                #x_total.append(x)
                n += 1
        intervals[i] = n
        i += 1
        n = 0
    return intervals#, x_total

def find_distribution(p):
    p_sorted = np.sort(p)
    amount = []
    x_axis = [0]
    for pi in p_sorted:
        if pi != x_axis[-1]:
            x_axis.append(pi)
            amount.append(1)
        else:
            amount[-1] += 1
    x_axis = x_axis[1:]
    return np.array(x_axis), np.array(amount)

# task a) plot histogram
def plot_hist(N):
    p = find_time_intervals(N)
    plt.hist(p,bins=50,density=True)
    plt.show()
#plot_hist(N)


# task b)
def plot_distr_and_curve(x,distr,alpha):
    x = x[:20]
    distr = distr[:20]
    plt.plot(x,distr)
    plt.plot(x,distr[0]*x**(-alpha))
    plt.show()
    
def plot_error(x,distr,alpha_values):
    error = np.zeros(len(alpha_values))
    for i in range(len(alpha_values)):
        y = distr[0]*x**(-alpha_values[i])
        error[i] = np.sum((distr-y)**2)
    plt.plot(alpha_values,error)  
    plt.show()

def find_alpha(x,distr):
    func = lambda x,alpha: distr[0]*x**(-alpha)
    param, param_cov = curve_fit(func,x,distr)
    return param[0]

def plot_alpha(N_list):
    alphas = np.zeros(len(N_list))
    for i in range(len(N_list)): 
        p = find_time_intervals(N_list[i])
        x, distr = find_distribution(p)
        alphas[i] = find_alpha(x,distr)
    print(alphas)
    plt.plot(N_list,alphas)
    plt.show()


#N = 500
#p = find_time_intervals(N)
#x, distr = find_distribution(p)
#plot_distr_and_curve(x,distr,1.27)  #shows a minimum around alpha=1.27, depends on N perhaps?
#alpha_values = [0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6]
#plot_error(x,distr,alpha_values)
#print(find_alpha(x,distr))  

#N_list = np.array([400,800,1600,3200])
#plot_alpha(N_list)

# Can either curve_fit the error curve for a given N and find the minimum of that function, or
# use curve_fit for the distr and find a converging number alpha for large N. Takes a long time.