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
#######################################
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc


############### 1 paket ##############
#fizikalne kostnante-globale spremenljivke:
w=0.2
lam = 10.
k = w**2
alp = k**(1/4)

#mreža:
N = 300
n = 10000 #velikost mreže v času
T = 2*np.pi/w
x = np.linspace(-40,40,N)
dx = np.abs(x[1] - x[0])

t = np.linspace(0, 10*T, n)
dt = t[1]-t[0]
print(dt)

psi_0 = zacetni_pogoj1(x)


def diag_H_1(N, dx, dt, x):
    potencial  = 0.5 * k * x**2
    b = 1j*dt/(2.*dx**2)
    a = -b/2.
    subdl = np.ones(N-1, dtype=complex)*a
    subdd = np.ones(N-1, dtype=complex)*a
    diag = np.ones(N, dtype=complex)*(1.+b+1j*dt/2 * potencial)

    return subdl, diag, subdd

def matrika_iz_diagonal(a,d,c):
    H = np.diag(a, k=-1) + np.diag(d) + np.diag(c, k=1)
    return H

def diag_H_2(N, dx, dt, x):
    b = 1j*dt/(2*dx**2)
    a = -b/2
    subdl = np.ones(N-1, dtype=complex)*a
    subdd = np.ones(N-1, dtype=complex)*a
    diag = np.ones(N, dtype=complex)*(1+b)

    return subdl, diag, subdd
print(diag_H_2(5, 2,-1,0))

def resitev_1(zacetni_pogoj, N, n, a,d,c, solver):
    res = np.zeros((N, n), dtype=complex)
    H = matrika_iz_diagonal(a,d,c)
    res[:,0] = zacetni_pogoj
    Hconj = np.conjugate(H)
    H1 = sps.csr_matrix(H)
    a_band = np.insert(a, 0, 0)
    c_band = np.insert(c, N-1, 0)
    H_banded = np.vstack((a_band, d))
    H_banded=np.vstack((H_banded, c_band))
    for i in range(1, n):
        if solver==1:
            res[:, i] = spsolve(H1, Hconj@(res[:, i-1]))
        elif solver==2:
            res[:,i] = np.linalg.solve(H, Hconj@(res[:, i-1]))
        elif solver==3:
            res[:,i] = TDMAsolver(a,d,c, Hconj@(res[:, i-1]))
        elif solver==4:
            res[:, i] = solve_banded((1,1),H_banded,  Hconj@(res[:, i-1]))
    return res

##### resitev prvi paket #########
#a,d,c = diag_H_1(N,dx,dt,x)
#resitev = resitev_1(psi_0, N,n,a,d,c, 4)


######################################################
####################### GRAFI 1 paket ########################
######################################################

plt.rcParams["axes.formatter.limits"]  = (-3,2)
def o_prvem_paketu():
    fig, ax = plt.subplots(3,2)
    for i in range(1,6):
        if i==5:
            i=n
            j = 10

        else:
            j = 2*i
            i*=2000
        ax[0][0].plot(x, np.abs(koherentno(x, t[0]))**2 -np.abs(koherentno(x, t[i-1]))**2 , label=r'q={}T'.format(j))
    for i in range(0,7):
        if i==6:
            i=n
            j = 10

        else:
            j = 2*i
            i*=2000
            ax[0][1].plot(x, np.abs(resitev[:,i-1])**2, label='{}T'.format(j))
    k=0
    for i in range(0, 100, 1):
        cmap = plt.get_cmap('jet',100)
        img = ax[1][0].plot(x, np.abs(np.abs(resitev[:,i])**2 - np.abs(koherentno(x, t[i]))**2) , c=cmap(k))
        k +=1
    norm = mpl.colors.Normalize(vmin=0,vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm,ax = ax[1][0], ticks=np.linspace(0,100*dt,10),
                 boundaries=np.arange(-0.05,2.1,.1))
    cbar.set_label('t', rotation=360)

    err = np.zeros((50, N))
    stevilo_tock = np.linspace(100,10000,50) #### 10000 j
    for k, i in enumerate(stevilo_tock):
        cmap = plt.get_cmap('jet',100)
        #print(i)
        t1 = np.linspace(0, 10*T, int(i))
        dt1 = t1[1]-t1[0]
        a1,d1,c1 = diag_H_1(N,dx,dt1,x)
        resitev1 = resitev_1(psi_0, N,int(i),a1,d1,c1, 4)
        #a = np.array(np.where(abs(t1- 31.4)< 0.022))

        #print(np.where(abs(t1- 31.4)< 0.022))
        err[k,:] = np.abs(np.abs(resitev1[:,-1])**2  - np.abs(koherentno(x, 0)**2))
        #plt.plot(x,np.abs(resitev[:,int(i/10)]**2 ))
        #plt.plot(x, np.abs(koherentno(x, 0)**2))
        #plt.show()
        #plt.clf()


    c = ax[1][1].contourf(x,stevilo_tock,err)
    ax[1][1].set_xlim((-2,15))

    cbar = fig.colorbar(c,ax = ax[1][1], orientation='vertical')
    cbar.set_label(r'$||\psi(10T)|^2-|\psi_{num}(10T)|^2|$')

    ######casovna zaht#####
    csps = np.array([])
    thomas= np.array([])
    banded = np.array([])
    linalg = np.array([])
    interval = np.array([])

    for i in range(10, 1000, 10):
        cas = 500
        t1 = np.linspace(0, 10*T, cas)
        x1 = np.linspace(-40,40,i)
        psi_1 = zacetni_pogoj1(x1)
        dx1 = np.abs(x[1] - x[0])
        dt1 = t1[1]-t1[0]
        a1,d1,c1 = diag_H_1(i,dx1,dt1,x1)
        stsps = timer()
        resitev_1(psi_1, i,cas,a1,d1,c1, 1)
        stopsps = timer()
        csps = np.append(csps, stopsps-stsps)
        stthom = timer()
        if i<=500:
            resitev_1(psi_1, i,cas,a1,d1,c1, 3)
        else:
            None
        stopthom = timer()
        thomas = np.append(thomas, stopthom-stthom)
        strband = timer()
        resitev_1(psi_1, i,cas,a1,d1,c1, 4)
        stopband = timer()
        banded = np.append(banded, stopband-strband)
        stlinal = timer()
        if i<=250:
            resitev_1(psi_1, i,cas,a1,d1,c1, 2)
        else:
            None
        stoplin = timer()
        linalg = np.append(linalg, stoplin-stlinal)
        interval = np.append(interval, i)

    ####verjetnostna gotsota#####
    prava_vsota = np.array([])
    vsota_banded = np.array([])
    cmap = plt.get_cmap('viridis',10)
    y = 0
    casovni_int = np.array([])
    for j in range(1000, 10001, 1000):
        t2 = np.linspace(0, 10*T, j)
        dt2 = t2[1]-t2[0]
        casovni_int = np.append(casovni_int, dt2)
        a2,d2,c2 = diag_H_1(N,dx,dt2,x)
        resitev = resitev_1(psi_0, N,j,a2,d2,c2, 4)
        for i in range(j):
            vsota_banded = np.append(vsota_banded, np.sum(np.abs(resitev[:, i])**2))
            if j==10000:
                prava_vsota = np.append(prava_vsota, np.sum(np.abs(koherentno(x, t2[i]))**2))
        vsota_banded -=np.sum(np.abs(koherentno(x, 0))**2)
        if j==10000:
            prava_vsota-= np.sum(np.abs(koherentno(x, 0))**2)
        ax[2][1].plot(t2,np.abs(vsota_banded), c =cmap(y) )
        vsota_banded = np.array([])
        y +=1

    ax[2][1].plot(t, prava_vsota, label='analiticna')
    #ax.matshow(newcm, cmap=pyplot.cm.Greys)
    norm = mpl.colors.Normalize(vmin=np.min(casovni_int),vmax=np.max(casovni_int))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm,ax = ax[2][1])
    cbar.set_label(r'$dt$', rotation=360)
    ax[2][1].set_xlabel('t')
    ax[2][1].legend(loc='lower left')
    ax[2][1].set_ylabel(r'$|PDF-PDF_{0}|$')

    ax[2][0].plot(interval, banded, label='solve_banded')
    ax[2][0].plot(interval[:25], linalg[:25], label='linalg.solve')
    ax[2][0].plot(interval[:50], thomas[:50], label='Thomas')
    ax[2][0].plot(interval, csps, label='spsolve')
    ax[2][0].set_ylabel(r'$t$')
    ax[2][0].legend(loc='upper right')
    ax[2][0].text(3, 10, r'$n_{t} = 500$')
    ax[2][0].set_xlabel(r'$n_{x}$')




    ax[0][1].plot(x, np.abs(koherentno(x, t[0]))**2, label=r'$\psi(0)$')
    ax[0][0].set_xlim((-10, 20))
    ax[0][0].set_xlabel('x')
    ax[0][0].set_ylabel(r'$|\psi(0)|^2-|\psi(q)|^2$')
    ax[0][0].text(-9, 0.9e-5, 'dt={:.3f}'.format(dt) )
    ax[0][0].legend(loc ='lower left')
    ax[0][1].legend(loc ='upper left')
    ax[0][1].set_xlim((-5, 15))
    ax[0][1].set_ylabel(r'|$\psi_{num}(nT)|^2$')
    ax[0][1].set_xlabel('x')
    ax[0][1].text(2, 0.37, 'dt={:.3f}'.format(dt) )
    ax[1][0].set_yscale('log')
    ax[1][0].set_ylim((1e-10, 1))
    ax[1][0].set_xlim((0, 15))
    ax[1][0].set_xlabel('x')
    ax[1][0].set_ylabel(r'$||\psi(t)|^2-|\psi_{num}(t)|^2|$')
    ax[1][0].text(2, 1e-2, 'dt={:.3f}'.format(dt) )
    ax[1][1].set_xlim((0, 15))
    #ax[1][1].set_yscale('log')
    #ax[1][1].set_ylim((1e-10, 1))
    ax[1][1].set_ylabel(r'$n_{t}$')
    ax[1][1].set_xlabel('x')





    plt.subplots_adjust(wspace=0.4)
    plt.legend()

    plt.show()
    return None

########################################
########## Drugi paket #################
##########################################


def o_gauss_paketu():
    sigma0 = 1/20
    k0 = 50*np.pi
    lamb =0.25
    N2 = 600
    x2 = np.linspace(-0.5, 1.5, N2)
    dx2 = x2[1] - x2[0]
    dt2 = 2*dx2**2
    t2 = np.arange(0,0.0032 + dt2, dt2 )
    n2 = len(t2)
    print(t2[-1])

    psi_0 = zacetni_gauss(x2)

    a,d,c = diag_H_2(N2,dx2,dt2,x2)
    resitev = resitev_1(psi_0, N2,n2,a,d,c, 4)

    fig, ax = plt.subplots(3,2)
    for k,i in enumerate(t2, 0):
        cmap = plt.get_cmap('viridis',n2)
        ax[0][0].plot(x2, np.real(analiticna_gauss(x2, i)),c=cmap(k))
        cmap1 = plt.get_cmap('jet',n2)
        ax[0][1].plot(x2, np.real(resitev[:, k]), c=cmap1(k))
        #ax[1][0].plot(x2, np.real(resitev[:,50]))
        #ax[1][0].plot(x2,np.real(analiticna_gauss(x2, t2[50])))

    #plt.plot(x2, resitev[:, 0])


    N4 = 2000
    x4 = np.linspace(-0.5, 1.5, N4)
    dx4 = x4[1] - x4[0]
    dt4 = 2*dx4**2
    t4 = np.arange(0,0.0032 + dt4, dt4 )
    n4 = len(t4)
    psi_4 = zacetni_gauss(x4)

    a4,d4,c4 = diag_H_2(N4,dx4,dt4,x4)
    resitev4 = resitev_1(psi_4, N4,n4,a4,d4,c4, 4)

    razlike_manjsikorak = np.zeros((N4, n4))
    razlike = np.zeros((N2, n2))

    for i in range(n2):
        razlike[:, i] = np.abs(np.real(resitev[:,i])-np.real(analiticna_gauss(x2, t2[i])))
    for i in range(n4):
        razlike_manjsikorak[:, i]=np.abs(np.real(resitev4[:,i])-np.real(analiticna_gauss(x4, t4[i])))

    c = ax[1][0].contourf(x2,t2,razlike.T)
    e = ax[2][1].contourf(x4, t4, razlike_manjsikorak.T)

    x_zamiki = np.array([])
    for j in range(400, 1001, 200):
        N3 = j
        x3 = np.linspace(-0.5, 1.5, N3)
        dx3 = x3[1] - x3[0]
        dt3 = 2*dx3**2
        t3 = np.arange(0,0.0032 + dt3, dt3 )
        n3 = len(t3)
        psi_1 = zacetni_gauss(x3)

        a1,d1,c1 = diag_H_2(N3,dx3,dt3,x3)
        resitev1 = resitev_1(psi_1, N3,n3,a1,d1,c1, 4)
        for i in range(n3):
            delta_x = np.abs(x3[np.where(np.abs(resitev1[:,i])**2 ==np.max(np.abs(resitev1[:,i])**2))] - x3[np.where(np.abs(analiticna_gauss(x3, t3[i]))**2 == np.max(np.abs(analiticna_gauss(x3, t3[i]))**2))])
            x_zamiki = np.append(x_zamiki, delta_x)
        ax[1][1].plot(t3, x_zamiki, label='Nt={}'.format(j))
        x_zamiki = np.array([])


    for i in range(n2):
        delta_x = np.abs(x2[np.where(np.abs(resitev[:,i])**2 ==np.max(np.abs(resitev[:,i])**2))] - x2[np.where(np.abs(analiticna_gauss(x2, t2[i]))**2 == np.max(np.abs(analiticna_gauss(x2, t2[i]))**2))])
        x_zamiki = np.append(x_zamiki, delta_x)



    ax[1][1].legend()
    ax[1][1].set_xlabel('t')
    ax[1][1].set_ylabel(r'$\Delta x$')

    ax[2][0].plot(x2, np.abs(resitev[:,-1])**2, label=r'$|\psi_{num}(x)|^2$')
    ax[2][0].plot(x2, np.abs(analiticna_gauss(x2, t2[-1]))**2, label=r'$|\psi(x)|^2$')
    ax[2][0].plot(x2 + x_zamiki[-1], np.abs(resitev[:,-1])**2, label=r'$|\psi_{num}(x+\Delta x)|^2$')
    ax[2][0].legend(loc='lower left')
    ax[2][0].set_xlim((0.5,1.0))
    ax[2][0].set_xlabel('x')
    ax[2][0].set_ylabel(r'$|\psi(t= 32ms)|^2$')



    norm = mpl.colors.Normalize(vmin=0,vmax=0.0032)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm,ax = ax[0][0], ticks=np.linspace(0,n2*dt2,10),
                 boundaries=np.arange(0,0.0032,.0001))
    cbar.set_label('t', rotation=360)
    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
    sm1.set_array([])
    cbar = fig.colorbar(sm1,ax = ax[0][1], ticks=np.linspace(0,n2*dt2,10),
                 boundaries=np.arange(0,0.0032,.0001))
    cbar.set_label('t', rotation=360)


    cbar = fig.colorbar(c,ax = ax[1][0], ticks=np.linspace(-3,3,5))
    cbar.set_label(r'$|Re(\psi) - Re(\psi_{num})|$', rotation=90)
    cbar1 = fig.colorbar(e,ax = ax[2][1])
    cbar1.set_label(r'$|Re(\psi) - Re(\psi_{num})|$', rotation=90)



    #ax[1][0].set_yscale('log')
    #ax[1][0].set_ylim((1e-10, 5))
    ax[2][1].set_xlabel('x')
    ax[2][1].set_ylabel('t')
    ax[2][1].set_title('Nt=2000', fontsize=10)
    ax[2][1].set_xlim((0.0,1.0))

    ax[1][0].set_xlim((0,1))
    ax[1][0].set_ylabel('t')
    ax[1][0].set_xlabel('x')
    ax[0][0].set_ylabel(r'$Re(\psi)$')
    ax[0][0].set_xlabel('x')
    ax[0][1].set_xlabel('x')
    ax[0][1].set_ylabel(r'$Re(\psi_{num})$')


    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    #plt.show()
    plt.clf()
    return None
