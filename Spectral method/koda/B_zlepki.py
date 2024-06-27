import numpy as np
from numpy import fft
from diffeq_dodane import *
from tridiagonal import *
from scipy import linalg
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def B_zlepki(dx, k, x0):
    # k = -1,...,n+1
    if(x0 <= dx*(k-2)):
        return 0.
    elif(x0 <= dx*(k-1)):
        return (x0 - (k-2)*dx)**3. / (dx**3.)
    elif(x0 <= dx*k):
        return (1/dx**3)*(x0-dx*(k-2))**3 - (4/dx**3)*(x0-dx*(k-1))**3
    elif(x0 <= dx*(k+1)):
        return (1/dx**3)*(dx*(k+2) - x0)**3 - (4/dx**3)*(dx*(k+1) -x0)**3
    elif(x0 <= dx*(k+2)):
        return -(x0 - dx*(k+2))**3. / (dx**3.)
    else:
        return 0.


def gauss(x, D=0.3, sigma=0.2, a=1., O=100):
    return(O *np.exp(-(x-a/2)**2/sigma**2))

def res(N, M, cas, a, D):
    cas = cas
    tk = np.linspace(0., cas, M)
    xk = np.linspace(0., a, N)
    dx = a/N
    dt = cas/M
    # def mat_A(N):
    #     A = np.diag(4*np.ones(N)) + np.diag(1*np.ones(N-1), k=1) + np.diag(1*np.ones(N-1),k= - 1)
    #     return(A)
    # def mat_B(N):
    #
    #     B = (6*D/dx**2)*(np.diag(-2*np.ones(N)) + np.diag(1*np.ones(N-1), k=1) + np.diag(1*np.ones(N-1),k= - 1))
    #     return(B)
    # A = mat_A(N-2)

    A_d = 4*np.ones(N-2)
    A_dz = 1*np.ones(N-3)
    A_ds = 1*np.ones(N-3)
    C_d = 4+dt*(6*D/dx**2)*np.ones(N-2)
    C_dz = 1-dt*0.5*(6*D/dx**2)*np.ones(N-3)
    C_ds = 1-dt*0.5*(6*D/dx**2)*np.ones(N-3)
    D_d = np.diag(4-dt*(6*D/dx**2)*np.ones(N-2))
    D_dz = np.diag(1+dt*0.5*(6*D/dx**2)*np.ones(N-3), k=1)
    D_ds =  np.diag(1+dt*0.5*(6*D/dx**2)*np.ones(N-3),k= - 1)
    E = D_d + D_dz + D_ds
    factor(C_ds, C_d, C_dz)
    factor(A_ds, A_d, A_dz)
    H = np.zeros((M, N))
    #matrika C mora biti večja za 4 od A in B
    C = np.zeros((M, N+2))
    C[0,2:-2] = solve(A_ds, A_d, A_dz, gauss(xk)[1:-1])
    #C[0,2:-2] = linalg.inv(A)@gauss(xk[1:-1])
    for j in range(1, len(tk)):
        C[j,2:-2 ] = solve(C_ds,C_d,C_dz, E@C[j-1, 2:-2])
    for j in range(len(tk)):
        for i in range(len(xk)):
            v = 0
            for k in range(len(xk)+2):
                v += C[j,k]*B_zlepki(dx, k-1, xk[i] )
            H[j, i] = v
    return H
def res_scipy(N, M, cas, a, D):
    cas = cas
    tk = np.linspace(0., cas, M)
    xk = np.linspace(0., a, N)
    dx = a/N
    dt = cas/M
    def mat_A(N):
         A = np.diag(4*np.ones(N)) + np.diag(1*np.ones(N-1), k=1) + np.diag(1*np.ones(N-1),k= - 1)
         return(A)
    def matB(N):
         B = (6*D/dx**2)*(np.diag(-2*np.ones(N)) + np.diag(1*np.ones(N-1), k=1) + np.diag(1*np.ones(N-1),k= - 1))
         return(B)
    A = mat_A(N-2)

    # A_d = 4*np.ones(N-2)
    # A_dz = 1*np.ones(N-3)
    # A_ds = 1*np.ones(N-3)
    C_d = np.diag(4+dt*(6*D/dx**2)*np.ones(N-2))
    C_dz = np.diag(1-dt*0.5*(6*D/dx**2)*np.ones(N-3), k=1)
    C_ds = np.diag(1-dt*0.5*(6*D/dx**2)*np.ones(N-3),k= - 1)
    D_d = np.diag(4-dt*(6*D/dx**2)*np.ones(N-2))
    D_dz = np.diag(1+dt*0.5*(6*D/dx**2)*np.ones(N-3), k=1)
    D_ds =  np.diag(1+dt*0.5*(6*D/dx**2)*np.ones(N-3),k= - 1)
    K = C_d + C_dz + C_ds
    E = D_d + D_dz + D_ds
    #factor(C_ds, C_d, C_dz)
    #factor(A_ds, A_d, A_dz)
    H = np.zeros((M, N))
    #matrika C mora biti večja za 4 od A in B
    C = np.zeros((M, N+2))
    #C[0,2:-2] = solve(A_ds, A_d, A_dz, gauss(xk)[1:-1])
    C[0,2:-2] = linalg.inv(A)@gauss(xk[1:-1])
    for j in range(1, len(tk)):
        #C[j,2:-2 ] = solve(C_ds,C_d,C_dz, E@C[j-1, 2:-2])
        C[j,2:-2] = linalg.solve_triangular(K, E@ C[j-1, 2:-2])
    for j in range(len(tk)):
        for i in range(len(xk)):
            v = 0
            for k in range(len(xk)+2):
                v += C[j,k]*B_zlepki(dx, k-1, xk[i] )
            H[j, i] = v
    return H
