from metode_qr import *
from hamiltonke import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy import linalg as LA
from scipy.linalg import schur
from timeit import default_timer as timer
import scipy.linalg as lin
import matplotlib.animation as animation
from celluloid import Camera
import matplotlib.patches as mpatches
import scipy.special as spec
from scipy.special import factorial
'''
 #############ANIMACIJA####################
lamda = np.linspace(0,1.0,10)

fig = plt.figure()
camera = Camera(fig)
plt.yscale('log')
plt.ylabel('sigma(E(N))')
plt.xlabel(r'$\lambda$')
red_patch = mpatches.Patch(color='red', label='[q^2]^2')
blue_patch = mpatches.Patch(color='blue', label='[q^4]')
yellow_patch = mpatches.Patch(color = 'green', label = '[q]^4')
plt.legend(handles=[red_patch, blue_patch, yellow_patch])
for i in range(0,30):
    seznam_pri_en = np.array([])
    h2 = np.array([])
    h3 = np.array([])
    for j in lamda:
        H = hamiltonka(1,j,100)
        H2 = hamiltonka(2,j,100)
        H3 = hamiltonka(3,j,100)
        lambda_b, vektorji = LA.eigh(H)
        lambda_b2, vektorji2 = LA.eigh(H2)
        lambda_b3, vektorji3 = LA.eigh(H3)
        h2 = np.append(h2, lambda_b2[i])
        h3 = np.append(h3, lambda_b3[i])
        seznam_pri_en = np.append(seznam_pri_en, lambda_b[i])

    prava = (h2 + h3 + seznam_pri_en)/3.
    plt.plot(lamda, abs(h2-prava), 'r')
    plt.plot(lamda, abs(h3-prava), 'b')
    plt.plot(lamda, abs(seznam_pri_en-prava) ,color='green')
    plt.yscale('log')
    plt.ylabel(r'$log(\sigma_{(E(N)))}$')
    plt.xlabel(r'$\lambda$')
    plt.title(r'')
    camera.snap()


#plt.title('$[q]^{4}$')
#plt.show()
#plt.pause(0.01)
#plt.clf()

animation = camera.animate()
animation.save('razlike_q_3.gif', writer='Pillow', fps=2)
###############################################3
'''

def primerjava(stevilo):
    H = hamiltonka(3,0.5,1000)
    H_10 = hamiltonka(stevilo,0.5,100)
    H_20 = hamiltonka(stevilo,0.5,100)
    H_50 = hamiltonka(stevilo,0.5,100)
    H_100 = hamiltonka(stevilo,1,100)
    lastne = LA.eigh(H)[0]
    lastne_10 = LA.eigh(H_10)[0]
    lastne_20 = LA.eigh(H_20)[0]
    lastne_50 = LA.eigh(H_50)[0]
    lastne_100 = LA.eigh(H_100)[0]
    h_10 = abs(lastne_10[:20] - lastne[:20])
    h_20 = abs(lastne_20[:20] - lastne[:20])
    h_50 = abs(lastne_50[:20] - lastne[:20])
    h_100 = abs(lastne_100 - lastne[:100])
    return(h_10, h_20, h_50, h_100)
plt.plot([i for i in range(20)], primerjava(1)[0], '-o', markersize = '2', label='H(10)')
plt.plot([i for i in range(20)], primerjava(2)[1], '-o', markersize = '2', label = 'H(20)')
plt.plot([i for i in range(20)], primerjava(3)[2], '-o', markersize = '2', label = 'H(50)')
#plt.plot([i for i in range(100)], primerjava(3)[3], '-o', markersize = '2', label= 'H(100)')
#plt.yscale('log')
#plt.ylabel(r'$log(\sigma({E_{prava} - E_{i}))}$')
#plt.xlabel(r'$N$')
#plt.title(r'$[q^{4}], \lambda = 1$')
#plt.legend()
plt.show()
plt.clf()

red_patch = mpatches.Patch(color='red', label=r'$\lambda = 0.75$')
blue_patch = mpatches.Patch(color='blue', label=r'$\lambda = 0.0$')
orange_patch = mpatches.Patch(color = 'orange', label = r'$\lambda = 0.25$')
green = mpatches.Patch(color = 'green', label = r'$\lambda = 0.5$')
viola = mpatches.Patch(color = 'purple', label = r'$\lambda = 1.0$')
plt.legend(handles=[viola,red_patch,green,orange_patch, blue_patch])
lamda = np.linspace(0,1.0,5)
#print(lamda)
for i in lamda:
    H = hamiltonka(2,i,1000)
    lastne = LA.eigh(H)[0]
    plt.plot([i for i in range(100)], lastne[:100], '-o', markersize='2')
plt.xlabel('N')
plt.ylabel('E(N)')
plt.title('$[q^{2}]^2$')
#plt.show()
plt.clf()
