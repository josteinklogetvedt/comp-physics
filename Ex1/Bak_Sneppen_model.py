import numpy as np
import matplotlib.pyplot as plt


def run_model(N,M):
    param = np.random.rand(M)
    x = np.zeros(N)
    y = np.arange(N)
    
    for i in range(N):
        w = np.argmin(param)
        if w == M-1:
            param[w-1], param[w], param[0] = np.random.rand(3)[:]
        else:
            param[w-1], param[w], param[w+1] = np.random.rand(3)[:]
        x[i] = w + 1 
    return x,y

def make_plot():
    N = 10000
    M = 1000
    x,y = run_model(N,M)
    
    size = np.ones(N)*1.5
    plt.rcParams["figure.figsize"] = (8,15)
    plt.scatter(x,y, s=size, color='black')
    plt.show()

make_plot()
