import numpy as np
import matplotlib.pyplot as plt
from cmath import *
import cmath
from matplotlib import cm
from analiticna_resitev import *
from tridiagonal import *
from analiticna_resitev import*
from scipy.sparse.linalg import spsolve
import scipy.sparse as sps
from matplotlib import ticker, cm
from scipy.linalg import solve_banded
import matplotlib as mpl
from timeit import default_timer as timer


####koeficienti v krajevnem delu, r=red, prvi element:k=0, drugi element:k=1 , k je obdiagonala
r1 = np.array([-2, 1])
r2 = np.array([-5/2, 4/3, -1/12])
r3 = np.array([-49/18, 3/2, -3/20, 1/90])
r4 = np.array([-205/72, 8/5, -1/5, 8/315,-1/560 ])
r5 = np.array([-5269/1800, 5/3, - 5/21, 5/126, - 5/1008,1/315])
r6 = np.array([-5369/1800, 12/7, -15/56, 10/189, - 1/112,2/1925, - 1/16632])
r7 = np.array([-266681/88200,7/4, - 7/24,7/108, - 7/528,7/3300, - 7/30888,1/84084])
##############koeficienti v padejevi approx (casovni del), m= red, prvi element:s=1, st elementov
############## je stevilo iteracij do naslednjega koraka
m1 = np.array([-2.0])
m2 = np.array([-3.0 + 1j*1.73205, -3.0 - 1j*1.73205])
m3 = np.array([-4.64437 , -3.67781 - 1j*3.50876, -3.67781 + 1j*3.50876])
m4 = np.array([-4.20758 + 1j*5.31484, -5.79242 + 1j*1.73447, -5.79242 - 1j*1.73446, -4.20758 -1j*5.31483])
m5 = np.array([-4.64935 + 1j*7.14205, -6.70391 + 1j*3.48532, -7.29348 + 1j*0.00000, -6.70391 - 1j*3.48532, -4.64935 - 1j*7.14205])



def matrika_banded(N, dx, dt, red_r, z):
    b = 1j*dt/(2*dx**2)
    diag_el = np.array([])
    for i in red_r:
        diag_el = np.append(diag_el, (b*i)/np.conjugate(z))
    d = 1 + diag_el[0]
    mat = np.zeros((len(diag_el),N), dtype=complex)
    mat[0, :] = np.ones(N)*d
    for i in range(1,len(diag_el)):
        mat[i,i:] = np.ones(N-i, dtype=complex)*diag_el[i]
    flip_mat = np.flip(mat, axis=0)
    res = np.vstack((flip_mat, np.flip(mat[1:,:], axis=1)))
    return res
def matrika(N, dx, dt, red_r, z):
    b = 1j*dt/(2*dx**2)
    diag_el = np.array([])
    for i in red_r:
        diag_el = np.append(diag_el, (b*i)/np.conjugate(z))
    d = 1 + diag_el[0]
    mat = np.diag(np.ones(N)*d)
    for i in range(1,len(diag_el)):
        mat += np.diag(np.ones(N-i)*diag_el[i], k=i)
        mat += np.diag(np.ones(N-i)*diag_el[i], k=-i)

    return(mat)





sigma0 = 1/20
k0 = 50*np.pi
lamb =0.25
N2 = 600
x2 = np.linspace(-0.5, 1.5, N2)
dx2 = x2[1] - x2[0]
dt2 = 2*dx2**2
t2 = np.arange(0,0.0032 + dt2, dt2 )
n2 = len(t2)
print(n2)

psi_0 = zacetni_gauss(x2)


def resitev_1(zacetni_pogoj, N, n, matrika_banded,matrika, st_sub):
    res = np.zeros((N, n), dtype=complex)
    res[:,0] = zacetni_pogoj
    H_banded = matrika_banded
    mat = matrika
    Hconj = np.conjugate(mat)
    for i in range(1, n):
            res[:, i] = solve_banded((st_sub,st_sub),H_banded, Hconj@(res[:, i-1]))
    return res

def padejeva(N, dx, dt, red_r, prejsnja_resitev, red_m, st_sub):
    for i in range(len(red_m)):
            mat = matrika(N, dx, dt, red_r, red_m[i])
            Hconj = np.conjugate(mat)
            matrika_band = matrika_banded(N, dx, dt, red_r, red_m[i])
            res = solve_banded((st_sub,st_sub),matrika_band, Hconj@ prejsnja_resitev)
            prejsnja_resitev=res
    return(res)

def resitev_2(zacetni_pogoj, N, n, st_sub, red_r, red_m):
    res = np.zeros((N, n), dtype=complex)
    res[:,0] = zacetni_pogoj
    for i in range(1, n):
            res[:, i] = padejeva(N, dx2, dt2, red_r, res[:, i-1],red_m, st_sub)
    return res

res = resitev_2(psi_0, N2, n2, 2, r2, m2)
err = np.zeros((N2, n2), dtype=complex)
for i in range(n2):
    err[:,i] = np.abs(np.real(res[:, i]) - np.real(analiticna_gauss(x2, t2[i])))

plt.rcParams["axes.formatter.limits"]  = (-3,2)
c = plt.contourf(x2,t2,err.T)
b = plt.colorbar(c, orientation='vertical')
b.set_label(r'|$Re(\psi_{num}) - Re(\psi)$|')
plt.xlabel('x')
plt.ylabel('t')
plt.title('r=7, m=5, Nx=1200')
plt.show()
#matrika_band = matrika_banded(N2, dx2, dt2, r4, -2.00)
#mat = matrika(N2,dx2,dt2,r4, -2.00)
#res = resitev_1(psi_0, N2, n2, matrika_band, mat, 4)

#for i in range(0, n2 ):
#    plt.plot(x2, np.real(res[:, 10]))
#    plt.plot(x2, np.real(analiticna_gauss(x2, t2[10])))
#plt.show()
