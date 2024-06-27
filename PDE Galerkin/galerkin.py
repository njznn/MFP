import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.linalg import block_diag
from scipy import linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import scipy.integrate as integrate

############################## ANALITICNA RESITEV#################

def koef_in_fje_od_n(n,st_nicel, r, fi):
    nicle = sc.jn_zeros(2*n+1,st_nicel)
    C_ns = np.array([])
    vec = np.array([])
    for i in range(st_nicel):
        koef = 4/(np.pi*(1+2*n)*nicle[i]**2)
        int = integrate.quad(lambda x: x*sc.jv(2*n+1, nicle[i]*x), 0, 1.0)[0]
        int2 = integrate.quad(lambda x: x*sc.jv(2*n+1, nicle[i]*x)**2, 0, 1.0)[0]
        C_ns = np.append(C_ns, koef*int/int2)
        vec = np.append(vec, sc.jv(2*n+1, nicle[i]*r)*np.sin((2*n+1)*fi))
    return C_ns, vec




def analiticna_v_tocki(n,s,r,fi):
    vrednost = 0.0
    for i in range(n):
        koef, vec = koef_in_fje_od_n(i,s, r, fi)
        vrednost += koef @ vec
    return vrednost

#print(analiticna_v_tocki(3,3,0.5,0.5))




#def analiticna_v_tocki2(n, s, r, fi):
#    vrednost = 0.0
#    for i in range(n):
#        koef = koef_in_fje_od_n(i,s, r, fi)[0]
#        nicle = sc.jn_zeros(2*i+1, s)
#        for j in range(s):
#            vrednost += koef[j]*sc.jv(2*i+1, nicle[j]*r)*np.sin((2*i+1)*fi)
#    return vrednost
#print(analiticna_v_tocki2(3,3,0.5,0.5))




####################
def bazne_funkcije(ksi, fi, m,n):
    return (ksi**(2*m+1))*(1-ksi)**n * np.sin((2*m+1)*fi)

def vektor_b(m,n):
    vec = np.array([])
    for i in range(m+1):
        vektor_n = np.array([])
        for j in range(1,n+1):
            vektor_n = np.append(vektor_n, (- 2/(2*i+1))*sc.beta(2*i+3,j+1))
        vec = np.append(vec, vektor_n)
    return vec

def blocna_mat(m, n):
    blok = np.zeros((n,n))
    for i in range(1,n+1):
        for j in range(1,n+1):
            blok[i-1][j-1] = (-np.pi/2)*sc.beta(i+j-1,3+4*m)*(i*j*(3+4*m))/(2+4*m+j+i)
    return blok


def matrika(m,n):
    mat = blocna_mat(0,n)
    if m >=1:
        for i in range(1,m+1):
            mat = block_diag(mat, blocna_mat(i,n))
    return mat

def koeficient(m,n):
    matrika1 = csc_matrix(matrika(m,n), dtype=float)
    C = (-32/np.pi)*vektor_b(m,n)@spsolve(matrika1,vektor_b(m,n))
    return C

def vrednost_v_tocki(r, fi,m,n,a):
    vrednost = 0
    indeks=0
    for i in range(m):
        for j in range(1,n+1):
            vrednost += a[indeks]*bazne_funkcije(r, fi, i,j)
            indeks +=1
    return vrednost


#m koeficienti se zacnejo z 0, n pa z 1, tako npr m=10, n=10 ustreza 11 clenom v m in 10 v n !!!


##########################GRAFI#####################
plt.rcParams["axes.formatter.limits"]  = (-3,2)
def hitrosti():
    N=50
    M=50
    a = spsolve(csc_matrix(matrika(10,10), dtype=float),vektor_b(10,10))
    r = np.linspace(0,1,N)
    fi = np.linspace(0, np.pi, M)
    R, FI = np.meshgrid(r, fi)
    X = R * np.cos(FI)
    Y = R * np.sin(FI)
    res = np.zeros((N,M))
    res_num = np.zeros((N,M))
    for j in range(M):
        res_num[:,j]=vrednost_v_tocki(r,fi[j],10,10, a)
    for i in range(N):
        for j in range(M):
            res[i][j] = analiticna_v_tocki(10,10,r[i],fi[j])
    kon = np.abs(res.T-res_num.T)
    #kon=res_num.T
    plt.contourf(R, FI,kon, np.linspace(0, kon.max(), 100))
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\xi$')
    plt.title('m=10,n=10,m*=10,s*=10')
    cbar = plt.colorbar()
    cbar.set_label(r'$|u-u_{num}|$')
    plt.show()
    return None

def konstanta():
    from matplotlib import ticker, cm
    import matplotlib as mpl
    ######velikosti m in n########
    m = 65
    n = 65
    xos = np.linspace(0,m,m+1)
    yos = np.linspace(1,n,n)
    referencna = 0.757721876571637 #koeficient(100,100)
    koeficienti = np.zeros((m+1,n), dtype=float)
    for i in range(0,m+1):
        for j in range(1,n+1):
            koeficienti[i,j-1] = np.abs(koeficient(i,j)-referencna)
    x, y = np.meshgrid(xos, yos)
    plt.contourf(x,y,koeficienti.T, norm=mpl.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label(r'$|C_{ref}-C_{m_{max}n_{max}}|$')
    plt.xlabel('m')
    plt.ylabel('n')

    plt.show()



    return None
konstanta()

def casovna_zahtevnost():
    from timeit import default_timer as timer
    for i in range(0,12,4):
        casi_num = np.array([])
        casi_an = np.array([])
        for j in range(1,15):
            zacnum = timer()
            a = spsolve(csc_matrix(matrika(i,j), dtype=float),vektor_b(i,j))
            vrednost_v_tocki(0.5,0.5, i, j, a)
            koncnum = timer()
            casi_num=np.append(casi_num, koncnum-zacnum)
            zacan = timer()
            analiticna_v_tocki(i,j,0.5,0.5)
            konan =timer()
            casi_an=np.append(casi_an, konan-zacan)
        plt.plot([k for k in range(1,15)], casi_num, label='numerična:m={}'.format(i))
        plt.plot([k for k in range(1,15)], casi_an, label='analitična:m={}'.format(i))
    plt.yscale('log')
    plt.xlabel('n')
    plt.title(r'$r=0.5, \xi=0.5$')
    plt.ylabel('t')
    plt.legend()
    plt.show()
#casovna_zahtevnost()
