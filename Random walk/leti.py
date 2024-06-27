import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.optimize import curve_fit
from scipy import optimize

np.random.seed(16)
#korak je sorazmeren s casom, npr 1 korak - 1 sekunda


def naslednji_korak(x,y):
    ro0 = np.random.uniform(0,1)
    fi = 2*np.pi*ro0
    a, m = 1, 2.
    l =(np.random.pareto(a) + 1) * m
    x += l*np.cos(fi)
    y += l*np.sin(fi)
    return(x,y, l)

def izracun_N_korakov_1_delec(st_korakov):
    x_kord = np.array([0])
    y_kord = np.array([0])
    l = np.array([0])
    for i in range(st_korakov-1):
        x_kord = np.append(x_kord, naslednji_korak(x_kord[-1], y_kord[-1])[0])
        y_kord = np.append(y_kord, naslednji_korak(x_kord[-1], y_kord[-1])[1])
        l = np.append(l, naslednji_korak(x_kord[-1], y_kord[-1])[2])
    return(x_kord, y_kord, l)

def izracun_N_delcev(st_delcev,st_korakov):
    vsi_x = np.empty((0,st_korakov))
    vsi_y = np.empty((0,st_korakov))
    vsi_l = np.empty((0,st_korakov))
    for i in range(st_delcev):
        A = izracun_N_korakov_1_delec(st_korakov)
        vsi_x = np.vstack([vsi_x, A[0]])
        vsi_y = np.vstack([vsi_y, A[1]])
        vsi_l = np.vstack([vsi_l, A[2]])
    return(vsi_x, vsi_y, vsi_l)


A = izracun_N_delcev(500,500)
#for i in range(7): #st_delcev
#    plt.plot(A[0][i], A[1][i])
#plt.xlabel('x')
#plt.ylabel('y')
plt.title('b) $\mu = 2$')
#plt.show()
###############################################33333
#ob casih izracunamo mad:
med_N = np.array([])
for j in range(1, 500): #stevilo korakov-1
        med = [(A[0][i][j]**2 + A[1][i][j]**2)**0.5 for i in range(500)]#stevilo delcev
        med_N = np.append(med_N, stat.median_abs_deviation(med))

def f(x,a,b):
    return(a*x+b)

log_x = np.array([np.log(i) for i in range(1, 500)])
plt.plot(log_x, 2*np.log(med_N), 'ro', label='točke', markersize = '2')
koef, kovarianca = optimize.curve_fit(f, log_x, 2*np.log(med_N))
plt.plot(log_x, f(log_x, *koef), label = 'fit')
print(koef[0], np.sqrt(np.diag(kovarianca))[0])
plt.xlabel('ln(t)')
plt.ylabel('2ln(MED)')
plt.legend()
plt.show()
