import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy import optimize
np.random.seed(12)


hitrost = 1
def naslednji_korak(x,y,hitrost):
    ro0 = np.random.uniform(0,1)
    fi = 2*np.pi*ro0
    a, m = 2.5, 2.
    l =(np.random.pareto(a) + 1) * m
    x += l*np.cos(fi)
    y += l*np.sin(fi)
    t = l/hitrost
    return(x,y,t)


def interpolacija(cas, prejsnji_x, prejsnji_y,x,y, prejsnji_t, t):
    inter_x = 0
    inter_y = 0
    if prejsnji_t < cas < t:
        inter_x = prejsnji_x + (cas-prejsnji_t)*((x-prejsnji_x)/(t-prejsnji_t))
        inter_y = prejsnji_y + (cas-prejsnji_t)*((y-prejsnji_y)/(t-prejsnji_t))
        return(inter_x,inter_y)
    else:
            None



def izracun_N_korakov_1_delec(st_korakov):
    x_kord = np.array([0])
    y_kord = np.array([0])
    casi = np.array([0])
    for i in range(1,st_korakov):
        A = naslednji_korak(x_kord[-1], y_kord[-1],hitrost)
        x_kord = np.append(x_kord, A[0])
        y_kord = np.append(y_kord, A[1])
        casi = np.append(casi, casi[-1]+A[2])
    return(x_kord, y_kord, casi)

#plt.plot(izracun_N_korakov_1_delec(100)[0], izracun_N_korakov_1_delec(100)[1])
#plt.show()
#def vrednosti_do_casa(seznam_x, seznam_y, seznam_t, zeljen_cas):
#    for i in range(1, len(seznam_t)):
#        if seznam_t[i-1] < zeljen_cas  and zeljen_cas < seznam_t[i]:
#            A = interpolacija(zeljen_cas, seznam_x[i-1], seznam_y[i-1], seznam_x[i], seznam_y[i],seznam_t[i-1], seznam_t[i])
#            x = A[0]
#            y = A[1]
#            sez_x = seznam_x[:i]
#            sez_x = np.append(sez_x, x)
#            sez_y = seznam_y[:i]
#            sez_y = np.append(sez_y, y)
#            return(sez_x, sez_y, seznam_t[:i])
#        elif seznam_t[i-1] == zeljen_cas:
#            return(seznam_x[:i], seznam_y[i], seznam_t[:i])
#        elif seznam_t[i] == zeljen_cas:
#            return(seznam_x[:i], seznam_y[:i], seznam_t[:i])


def izracun_N_delcev(st_delcev,st_korakov):
    vsi_x = np.empty((0,st_korakov))
    vsi_y = np.empty((0,st_korakov))
    vsi_casi = np.empty((0,st_korakov))
    for i in range(st_delcev):
        A = izracun_N_korakov_1_delec(st_korakov)
        vsi_x = np.vstack([vsi_x, A[0]])
        vsi_y = np.vstack([vsi_y, A[1]])
        vsi_casi = np.vstack([vsi_casi, A[2]])
    return(vsi_x, vsi_y, vsi_casi)




D=izracun_N_delcev(100, 100)
for i in range(100): #st_delcev
    plt.plot(D[0][i], D[1][i])
    print(D[2][i][-1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('d) $\mu = 3.5$')
plt.legend()
plt.show()
##############################################################
#porazdelitev koordinat pri danem casu t oz.dolzini l:

def poisci_interpolacijo_poljubnega_casa(seznam_cas, seznam_x, seznam_y, cas):
    for i in range(1, len(seznam_cas)):
        inter_x = 0
        inter_y = 0
        if seznam_cas[i-1] < cas < seznam_cas[i]:
            inter_x = seznam_x[i-1] + (cas-seznam_cas[i-1])*((seznam_x[i]-seznam_x[i-1])/(seznam_cas[i]-seznam_cas[i-1]))
            inter_y = seznam_y[i-1] + (cas-seznam_cas[i-1])*((seznam_y[i]-seznam_y[i-1])/(seznam_cas[i]-seznam_cas[i-1]))
            return(inter_x,inter_y)
        elif seznam_cas[i-1] == cas:
            return(seznam_x[i-1], seznam_y[i-1])
        elif seznam_cas[i] == cas:
            return(seznam_x[i], seznam_y[i])
        else:
                None


med_N_x = np.array([])
med_N_y = np.array([])
casi = np.linspace(2, 500, 200)

for i in casi:
    interpolirani_x = np.array([])
    interpolirani_y = np.array([])
    for j in range(100): #stevilo delcev
        A = poisci_interpolacijo_poljubnega_casa(D[2][j], D[0][j], D[1][j], i)
        interpolirani_x = np.append(interpolirani_x, A[0])
        interpolirani_y = np.append(interpolirani_y, A[1])
    med_N_x = np.append(med_N_x, stat.median_abs_deviation(interpolirani_x))
    med_N_y = np.append(med_N_y, stat.median_abs_deviation(interpolirani_y))

l_cele = np.sqrt(np.add(med_N_x**2, med_N_y**2))

def f(x,a,b):
    return(a*x+b)
#plt.plot(np.log(casi), 2*np.log(med_N_x), 'ro')
plt.plot(np.log(casi), 2*np.log(l_cele), 'ro',label='tocke',  markersize = '2')
koef, kovarianca = optimize.curve_fit(f, np.log(casi), 2*np.log(l_cele))
#plt.plot(np.log(casi), f(np.log(casi), *koef), label='fit')
#plt.xlabel('ln(t)')
#plt.ylabel('2ln(MED)')
#plt.legend()
#print(np.sqrt(np.diag(kovarianca)), koef[0])
#plt.show()
