import numpy as np
import matplotlib.pyplot as plt

from classifiers import *

def plot_fractal(fractal,savefig=False):
    f = np.vstack((fractal,fractal[0,:]))
    plt.plot(f[:,0],f[:,1],color='black',linewidth=0.8)
    plt.grid()
    if savefig:
        plt.savefig(savefig)
    plt.show()
    
def plot_contour(U,xv,yv,l=False,savename=False):

    plt.contourf(xv,yv,U,cmap='seismic')
    plt.xlabel(r'$\frac{x}{L}$')
    plt.ylabel(r'$\frac{y}{L}$')
    if l:
        frac = Koch_fractal(l)
        f = np.vstack((frac,frac[0,:]))
        plt.plot(f[:,0],f[:,1],color='black',linewidth=0.6)
    if savename:
        plt.savefig(savename+".pdf")
    plt.show()

from scipy.optimize import curve_fit
def f(x,a,b):
    return a*x+b

def plot_d_N():
    w_22 = np.load('Data/w_22.npz')['arr_0']  # the first hundred eigenvalues
    w_23 = np.load('Data/w_23.npz')['arr_0'] 
    #w_33 = np.load('Data/w_33.npz')['arr_0']
    w_34 = np.load('Data/w_34.npz')['arr_0']
    w_44 = np.load('Data/w_44.npz')['arr_0']
    #w_45 = np.load('Data/w_45.npz')['arr_0']
    w_55 = np.load('Data/w_55.npz')['arr_0']
    W = [w_23,w_34,w_44,w_55]
    labels = [r'$(2,3)$',r'$(3,4)$',r'$(4,4)$',r'$(5,5)$']
    for i in range(len(W)):
        w = np.sort(np.sqrt(np.real(W[i])))
        d_N = w**2/(4*np.pi) - np.arange(0,len(w))
        popt, pcov = curve_fit(f,np.log10(w),np.log10(d_N))
        print('d=',popt[0])
        plt.plot(w,d_N,label=labels[i])

    plt.legend()
    plt.grid()
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Delta N(\omega)$')
    plt.xscale('log')
    plt.yscale('log')
    #plt.savefig('d_N_all.pdf')
    plt.show()