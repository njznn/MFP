import numpy as np
import matplotlib.pyplot as plt
from DFT import *
from numpy import fft
from diffeq_2 import *
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from B_zlepki import *

'''
#interval :
N = 1000
M = 1000
#konstante:
a = 1.
D=0.3
cas = 0.5

tk = np.linspace(0., cas, M)
xk = np.linspace(0., a, N)
dx = a/N
dt = cas/M
fv = 1/dx
print(fv)
nuc = 0.5/dx
'''


def gauss(x, D=0.3, sigma=0.2, a=1., C=200):
    return(C *np.exp(-(x-a/2)**2/sigma**2))
def sin(x):
    return np.sin(4*np.pi*x/1.)

def res_periodicni():
    a = 1.
    N = 5000
    M = 5000
    D=0.03
    cas = 1.0
    tk = np.linspace(0., 1.0, M)
    xk = np.linspace(0., 1.0, N)
    dx = a/N
    dt = cas/M
    fv = 1/dx
    print(fv)
    nuc = 0.5/dx
    #fx = gauss(xk)
    fx =sin(xk)
    Fk = fft.fft(fx)
    #Fk = fft.fftshift(Fk)
    #fk = fft.fftshift(fk)
    H = np.empty((M, N), dtype=complex)
    fk = fft.fftfreq(len(H[0, :]),dx)

    def f(x,t):
        return (-4*D*(np.pi * fk)**2)*x

    for i in range(len(xk)):
        H[:, i] = Fk[i] * np.exp(-4*D*(np.pi*fk[i])**2*tk)
    for s in range(len(tk)):
        H[s, :] = fft.ifft(H[s,:])
        #plt.plot(xk, np.real(H[s,:]))
    #plt.show()
    return np.real(H)


def res_direchlet():
    #####zaradi ohranitve energije dobimo pol manjso amplitudo ker razsirimo robni pogoj
    fx = gauss(xk)
    fx[0], fx[-1] = 0.,0.
    fx = np.append(fx, -fx[:0:-1])
    Fk = fft.fft(fx)
    #Fk = fft.fftshift(Fk)
    #fk = fft.fftshift(fk)
    H = np.empty((M, 2*N-1), dtype=complex)
    fk = fft.fftfreq(len(H[0, :]),dx)


    def f(x,t):
        return(D*(-4*np.pi**2 * fk**2)*x)
    #Tk_(t)
    for i in range(len(xk)):
        H[:, i] = Fk[i] * np.exp(-4*D*(np.pi*fk[i])**2*tk)
    for s in range(len(tk)):
        H[s, :] = fft.ifft(H[s,:])
        #plt.plot(xk, H[s,:len(xk)])
    return np.real(H[:,:len(xk)])


def res_direchlet_analiticna():
    a = 1.
    N = 5000
    M = 5000
    D=0.3
    cas = 1.0
    tk = np.linspace(0., 1.0, M)
    xk = np.linspace(0., 1.0, N)
    dx = a/N
    dt = cas/M
    fv = 1/dx
    print(fv)
    nuc = 0.5/dx
    #####zaradi ohranitve energije dobimo pol manjso amplitudo ker razsirimo robni pogoj
    fx = gauss(xk)
    fx[0], fx[-1] = 0.,0.
    fx = np.append(fx, -fx[:0:-1])
    Fk = fft.fft(fx)
    #Fk = fft.fftshift(Fk)
    #fk = fft.fftshift(fk)
    H = np.empty((M, 2*N-1), dtype=complex)
    fk = fft.fftfreq(len(H[0, :]),dx)


    def f(x,t):
        return(D*(-4*np.pi**2 * fk**2)*x)
    #Tk_(t)
    for i in range(len(xk)):
        H[:, i] = Fk[i] * np.exp(-4*D*(np.pi*fk[i])**2*tk)
    for s in range(len(tk)):
        H[s, :] = fft.ifft(H[s,:])
        #plt.plot(xk, H[s,:len(xk)])
    return np.real(H[:,:len(xk)])
#######pazi na pogoj o stabilnosti eulerja########

def res_periodicni_euler():
    #M = 10000
    #N = 100
    #tk = np.linspace(0., 0.5, M)
    #xk = np.linspace(0., 1, N)
    #dt = 0.5/M
    #dx = a/N
    fx = gauss(xk)
    Fk = fft.fft(fx)
    #Fk = fft.fftshift(Fk)
    #fk = fft.fftshift(fk)
    H = np.empty((M, N), dtype=complex)
    fk = fft.fftfreq(len(H[0, :]),dx)
    def f(x,t):
        return (-4*D*(np.pi * fk[i])**2)*x

    for i in range(len(xk)):
        #print(1>abs(1+dt*D*(-4*np.pi**2 * fk[i]**2)))
        H[:,i] = euler(f,Fk[i], tk)
    for s in range(len(tk)):
        H[s, :] = fft.ifft(H[s,:])
    return np.real(H)

def res_direchlet_euler():
    M = 15000
    N = 100
    tk = np.linspace(0., 0.5, M)
    xk = np.linspace(0., 1, N)
    dt = 0.5/M
    dx = a/N
    fx = gauss(xk)
    fx[0], fx[-1] = 0.,0.
    fx = np.append(fx, -fx[:0:-1])
    Fk = fft.fft(fx)
    #Fk = fft.fftshift(Fk)
    #fk = fft.fftshift(fk)
    H = np.empty((M, 2*N-1), dtype=complex)
    fk = fft.fftfreq(len(H[0, :]),dx)
    def f(x,t):
        return (-4*D*(np.pi * fk[i])**2)*x

    for i in range(len(xk)):
        print(1>abs(1+dt*D*(-4*np.pi**2 * fk[i]**2)))
        print(fk[i])
        H[:,i] = euler(f,Fk[i], tk)
    for s in range(len(tk)):
        H[s, :] = fft.ifft(H[s,:])
        #plt.plot(xk, np.real(H[s,:len(xk)]))
    return np.real(H[:, :len(xk)])
#res_direchlet_euler()
######casovna
from timeit import default_timer as timer

def stabilnost_in_casovna_zaht():
    Nt = np.arange(10,151,10)
    Nx = np.arange(10,151,10)
    C = np.zeros((len(Nt), len(Nx)))
    B = np.zeros((len(Nt), len(Nx)))
    for k, i in enumerate(Nt,0):
        for l,j in enumerate(Nx,0):
            # N = j
            # M = i
            # a = 1.
            # D=0.2
            # cas = 0.5
            # tk = np.linspace(0., cas, M)
            # xk = np.linspace(0., a, N)
            # dx = a/N
            # dt = cas/M
            # nuc = 0.5/dx
            #if (1>abs(1+dt*D*(-4*np.pi**2 * nuc**2))):
            #st1 = timer()
            #res(j,i, 1., 0.5, 0.2)
            #res_scipy(j,i, 1., 0.5, 0.2)
            #res_periodicni_euler()
            #stop1 = timer()
            st = timer()
            res(j,i, 1., 0.5, 0.2)
            #res_scipy(j,i, 1., 0.5, 0.2)
            #res_periodicni_euler()
            stop = timer()
            C[k,l] = stop-st
            #B[k,l] = stop1-st1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X,Y = np.meshgrid(Nx,Nt)
    ax.plot_surface(X, Y, C-B, cmap='viridis',
                         linewidth=0, antialiased=False)

    #plt.clim(0.5, np.max(C))
    #fig.colorbar(surf, shrink=0.5, aspect=20, label=r'$\Delta \tau$')
    fig.tight_layout()
    #ax.set_zticks([])
    ax.set_zlabel(r'$\tau_{impl}$')
    ax.set_xlabel(r'$N_{x}$')
    ax.set_ylabel(r'$N_{t}$')
    #ax.set_title(r'')
    plt.show()
    return None
#stabilnost_in_casovna_zaht()
def res_direchlet_dodatna():
    a = 1.
    N = 5000
    M = 5000
    D=0.5
    cas = 1.0
    tk = np.linspace(0., 1.0, M)
    xk = np.linspace(0., 1.0, N)
    dx = a/N
    dt = cas/M
    fv = 1/dx
    print(fv)
    nuc = 0.5/dx
    #####zaradi ohranitve energije dobimo pol manjso amplitudo ker razsirimo robni pogoj
    fx = 2*sin(xk)
    fx[0], fx[-1] = 0.,0.
    fx = np.append(fx, -fx[:0:-1])
    Fk = fft.fft(fx)
    #Fk = fft.fftshift(Fk)
    #fk = fft.fftshift(fk)
    H = np.empty((M, 2*N-1), dtype=complex)
    fk = fft.fftfreq(len(H[0, :]),dx)


    def f(x,t):
        return(D*(-4*np.pi**2 * fk**2)*x)
    #Tk_(t)
    for i in range(len(xk)):
        H[:, i] = Fk[i] * np.exp(-4*D*(np.pi*fk[i])**2*tk)
    for s in range(len(tk)):
        H[s, :] = fft.ifft(H[s,:])
        #plt.plot(xk, H[s,:len(xk)])
    return np.real(H[:,:len(xk)])
