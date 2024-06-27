import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import metrics
import data_higgs as dh
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

barve_temne = ListedColormap(['#ff4d4d', '#668cff'])
np.random.seed(5)

def kvadrati(test_size=1000, learn_size=10000):
    N = test_size+learn_size
    x = np.zeros((N,2))

    x[:,] = np.random.uniform(-1,1, (N,2))
    indeksi = np.array([])
    for vrstica in x:
        if vrstica[0] > 0 and vrstica[1] > 0 or vrstica[0] < 0 and vrstica[1] <0:
            indeksi = np.append(indeksi, 1)
        else:
            indeksi = np.append(indeksi, 0)
    test = x[:test_size]
    test_indeks = indeksi[:test_size]
    learn = x[test_size:]
    learn_indeks = indeksi[test_size:]
    return learn, learn_indeks,test, test_indeks
#train_set, train_labels, test_set, test_labels = kvadrati()



def kroga(train_size=10000, var1=0.03, var2=0.01, test_size=1000):


	N = train_size + test_size

	x = np.zeros((N, 2))
	x[:N//2] = np.random.multivariate_normal(mean=[0,0],cov=[[var1,0],[0,var1]], size=N//2)

	t = np.linspace(0, 2*np.pi, N//2)
	x[N//2:] = np.array([np.cos(t), np.sin(t)]).T
	x[N//2:] += np.random.multivariate_normal(mean=[0,0],cov=[[var2,0],[0,var2]], size=N//2)

	y = np.zeros(N)
	y[N//2:] = 1
	index = np.arange(N)
	np.random.shuffle(index)

	x = x[index]
	y = y[index]

	test_set = x[-test_size:]
	test_labels = y[-test_size:]
	train_set = x[:-test_size]
	train_labels = y[:-test_size]

	return train_set, train_labels, test_set, test_labels

#train_set, train_labels, test_set, test_labels = kroga()
def mnozici(train_size=10000, var1=0.1, var2=0.1, test_size=1000):
    N = train_size + test_size
    x = np.zeros((N, 2))
    x[:N//2] = np.random.multivariate_normal(mean=[1,1],cov=[[var1,0],[0,var1]], size=N//2)
    x[N//2:] = np.random.multivariate_normal(mean=[-1,-1],cov=[[var2,0],[0,var2]], size=N//2)

    y = np.zeros(N)
    y[N//2:] = 1
    index = np.arange(N)
    np.random.shuffle(index)

    x = x[index]
    y = y[index]

    test_set = x[-test_size:]
    test_labels = y[-test_size:]
    train_set = x[:-test_size]
    train_labels = y[:-test_size]

    return train_set, train_labels, test_set, test_labels

def spirale(test_size=1000, learn_size=10000, turns=5):
    N = learn_size + test_size

    spiral = np.zeros((N, 2))
    spiral_index = [-1 for i in range(N//2)] + [1 for i in range(N//2)]

    spiral_index = np.array(spiral_index)

    np.random.shuffle(spiral_index)

    s = np.linspace(0,turns*np.pi, N)
    spiral = spiral_index * [s * np.cos(s), s * np.sin(s)]
    spiral = spiral.transpose()

    spiral_index[spiral_index == -1] = 0
    spiral += np.random.multivariate_normal(mean=[0,0],cov=[[0.1,0],[0,0.1]],size=N)
    spiral = spiral / 10

    index = np.arange(N)
    np.random.shuffle(index)

    spiral = spiral[index]
    spiral_index = spiral_index[index]

    train_set = spiral[:-test_size]
    test_set = spiral[-test_size:]
    train_labels = spiral_index[:-test_size]
    test_labels = spiral_index[-test_size:]
    return train_set, train_labels, test_set, test_labels
#train_set, train_labels, test_set, test_labels = spirale()
#plt.scatter(train_set[:, 0], train_set[:, 1], c=train_labels, cmap=barve_temne)
#plt.show()
