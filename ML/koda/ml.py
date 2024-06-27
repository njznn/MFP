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
from matplotlib.colors import ListedColormap


def split_xy(rawdata):
    #split features and labels from data
    #prepare the data => normalizations !

    # split
    data_y=rawdata['hlabel'] # labels only: 0.=bkg, 1.=sig
    data_x=rawdata.drop(['hlabel'], axis=1) # features only

    #now prepare the data
    mu = data_x.mean()
    s = data_x.std()
    dmax = data_x.max()
    dmin = data_x.min()

    # normal/standard rescaling
    #data_x = (data_x - mu)/s

    # scaling to [-1,1] range
    #data_x = -1. + 2.*(data_x - dmin)/(dmax-dmin)

    # scaling to [0,1] range
    data_x = (data_x - dmin)/(dmax-dmin)


    return data_x,data_y

def plot_history(histories, key='binary_crossentropy'):
#def plot_history(histories, key='acc'):
    plt.figure(figsize=(6,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')
    plt.title("Učenje")
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.show()

podatki = dh.load_data('/home/ziga/Desktop/FMF/3.letnik/MFP/ML/')
imena_kategorij=podatki['feature_names'].to_numpy()[1:]

x_trn, y_trn = split_xy(podatki['train'])
x_train, x_test,y_train, y_test = train_test_split(x_trn,y_trn,test_size=0.1)
x_val,y_val=split_xy(podatki['valid']) # independent cross-valid sample
#print(y_val)

################################# (D)NN #############################
def naredi_model(st_per, velikost, early_stop_bool): #######st_per je array
    from tensorflow.keras.utils import plot_model

    model = Sequential()
    model.add(Dense(st_per[0], input_shape=(velikost,)))
    model.add(Activation('relu'))
    for i in st_per[1:]:
        model.add(Dense(i))
        model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid')) # # output layer/value
    #plot_model(model, to_file='dnn_model1.png', show_shapes=True)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC', 'binary_crossentropy'])
    if early_stop_bool== True:
        early_stop = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=5)
    else:
        early_stop= EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=50000)

    return model, early_stop

#model1 = naredi_model([32, 32], 18, False)


#ds_train = tf.data.Dataset.from_tensor_slices((x_train.to_numpy(),y_train.to_numpy()))
#history_data=model1[0].fit(epochs=100, batch_size=500,
#                x=x_train, y=y_train,
#                validation_data=(x_test,y_test), callbacks=[model1[1],])

#y_predicted=model1[0].predict(x_val.to_numpy())[:,0]


def plot_roc_od_epoch():
    epoch = np.array([10,20,30,50])
    for i in epoch:
        model1 = naredi_model([32, 32, 32,32,32,32,32,32,32,32,32,32,32,32], 18, True)
        history_data=model1[0].fit(epochs=i, batch_size=750,
                        x=x_train, y=y_train,
                        validation_data=(x_test,y_test), callbacks=[model1[1],])
        y_predicted=model1[0].predict(x_val.to_numpy())[:,0]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_predicted)
        AUC =  metrics.auc(fpr, tpr) ###ploščina
        plt.plot(fpr, tpr, label='{},AUC = {:.2f}'.format(i,AUC))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.text(0.6, 0.2, 'št. nevtronskih plasti:15\nšt.perceptronov v plasti:32', fontsize = 10)
    plt.title('ROC krivulja')
    plt.legend()
    plt.show()

#plot_roc_od_epoch()
def plot_roc_od_velikosti():
    velikost_vzorca = np.array([6000,12000,18000,24000,32000,36000])
    stevilo_delcev = np.array([8,10,12,14,16,18])
    for i in stevilo_delcev:
        x_trn, y_trn = split_xy(podatki['train'])
        x_trn = x_trn.iloc[:,:i]
        x_val,y_val=split_xy(podatki['valid'])
        x_val = x_val.iloc[:,:i]
        #y_tr = y_train.iloc[:i]
        x_train, x_test,y_train, y_test = train_test_split(x_trn,y_trn,test_size=0.1)
        #x_train = x_train.iloc[:,0:i+1]
        #y_train = y_train.iloc[:,0:i+1]
        model1 = naredi_model([32, 32, 32,32,32,32,32,32,32,32], i, True)
        history_data=model1[0].fit(epochs=30, batch_size=750,
                        x=x_train, y=y_train,
                        validation_data=(x_test,y_test), callbacks=[model1[1],])
        y_predicted=model1[0].predict(x_val.to_numpy())[:,0]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_predicted)
        AUC =  metrics.auc(fpr, tpr) ###ploščina
        plt.plot(fpr, tpr, label='{:.3f},AUC = {:.2f}'.format(i,AUC))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.text(0.6, 0.2, 'št. nevtronskih plasti:15\nšt.perceptronov v plasti:32', fontsize = 10)
    plt.title('ROC krivulja')
    plt.legend()
    plt.show()
#plot_roc_od_velikosti()

def plot_ucenje(history_data):
    #pd.DataFrame(history_data.history).plot(figsize=(8, 5))
    #plt.grid(True)
    #plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    plot_history([('DNN model', history_data),],key='binary_crossentropy')
    plot_history([('DNN model', history_data),],key='auc')
    plot_history([('DNN model', history_data),],key='accuracy')
    plt.show()
#plot_ucenje(history_data)
#plot_ucenje(history_data)
def AUC_od_percept_in_layer():
    stevilo_plasti = np.linspace(20,25,5)
    st_perceptronov = np.linspace(50,55,5)
    (X,Y) = np.meshgrid(len(stevilo_plasti),len(st_perceptronov))
    AUC = np.zeros((len(stevilo_plasti), len(st_perceptronov)))
    for i,k in enumerate(stevilo_plasti):
        for j,l in enumerate(st_perceptronov):
            seznam = np.ones(int(k))*l
            model1 = naredi_model(seznam, 18, False)
            history_data=model1[0].fit(epochs=10, batch_size=750,
                            x=x_train, y=y_train,
                            validation_data=(x_test,y_test), callbacks=[model1[1],])
            y_predicted=model1[0].predict(x_val.to_numpy())[:,0]
            fpr, tpr, thresholds = metrics.roc_curve(y_val, y_predicted)
            AUC[i][j] =  metrics.auc(fpr, tpr) ###ploščina
    c = plt.contourf(stevilo_plasti,st_perceptronov,AUC.T)
    b = plt.colorbar(c, orientation='vertical')
    b.set_label(r'$AUC$')
    plt.title(r'$batch=750, epochs=10$')
    plt.xlabel('število plasti nevronskih mrež')
    plt.ylabel('število perceptronov')
    #plt.plot(fpr, tpr)
    plt.show()
    return None
#AUC_od_percept_in_layer()


###################################################################
########################## ADA BOOST in GBRT ######################
###################################################################
def naredi_ada(max_globina, n_est, learning_rate):
      ada = AdaBoostClassifier(
      DecisionTreeClassifier(max_depth=max_globina), n_estimators=n_est,
      algorithm="SAMME.R", learning_rate=learning_rate)
      return ada

def plot_AUC_od_max_gl_in_n_est():
    m = 4
    n = 20
    globina = np.arange(1,m+1,1)
    n_est = np.arange(1,n+1, 1)
    (X,Y) = np.meshgrid(globina,n_est)
    AUC = np.zeros((m, n))
    for i in range(1,m+1):
        for j in range(1,n+1):
            model = naredi_ada(i, j, 0.1)
            model.fit(x_train, y_train)
            y_score = model.predict_proba(x_val)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score)
            AUC[i-1][j-1] =  metrics.auc(fpr, tpr) ###ploščina
    c = plt.contourf(globina,n_est,AUC.T)
    b = plt.colorbar(c, orientation='vertical')
    b.set_label(r'AUC')
    plt.xlabel('globina')
    plt.ylabel('število dreves')
    #plt.plot(fpr, tpr)
    plt.show()
    return None
#plot_AUC_od_max_gl_in_n_est()

def plot_AUC_od_ucenja_in_n_est():
    n = 10
    ucenje = np.linspace(0.1,2.0,10)
    n_est = np.arange(1,n+1, 1)
    (X,Y) = np.meshgrid(ucenje,n_est)
    AUC = np.zeros((len(ucenje), n))
    for indeks, vred in enumerate(ucenje):
        for j in range(1,n+1):
            model = naredi_ada(2, j, vred)
            model.fit(x_train, y_train)
            y_score = model.predict_proba(x_val)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score)
            AUC[indeks][j-1] =  metrics.auc(fpr, tpr) ###ploščina
    c = plt.contourf(ucenje,n_est,AUC.T)
    b = plt.colorbar(c, orientation='vertical')
    b.set_label(r'$AUC$')
    plt.title(r'$globina=2$')
    plt.xlabel('hitrost učenja')
    plt.ylabel('število dreves')
    #plt.plot(fpr, tpr)
    plt.show()
    return None
#plot_AUC_od_ucenja_in_n_est()

def plot_roc_od_globine():
    globina = 5
    for i in range(1,globina+1):
        model = naredi_ada(i, 32, 0.1)
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_val)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score)
        plt.plot(fpr, tpr, label='{}, AUC= {:.2f}'.format(i, metrics.auc(fpr, tpr)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC krivulja, ST=32')
    plt.legend()
    plt.show()
    return None
#plot_roc_od_globine()
def plot_roc_od_velikosti_ada():
    velikost_vzorca = np.array([6000,12000,18000,24000,32000,36000])
    stevilo_delcev = np.array([8,10,12,14,16,18])
    for i in stevilo_delcev:
        x_trn, y_trn = split_xy(podatki['train'])
        x_trn = x_trn.iloc[:,:i]
        x_val,y_val=split_xy(podatki['valid'])
        x_val = x_val.iloc[:,:i]
        #y_tr = y_train.iloc[:i]
        x_train, x_test,y_train, y_test = train_test_split(x_trn,y_trn,test_size=0.1)
        #x_train = x_train.iloc[:,0:i+1]
        #y_train = y_train.iloc[:,0:i+1]
        model = naredi_ada(3, 32, 0.1)
        model.fit(x_train, y_train)
        y_score = model.predict_proba(x_val)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score)
        AUC =  metrics.auc(fpr, tpr) ###ploščina
        plt.plot(fpr, tpr, label='{},AUC = {:.2f}'.format(i,AUC))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.text(0.6, 0.2, 'št. nevtronskih plasti:15\nšt.perceptronov v plasti:32', fontsize = 10)
    plt.title('ROC krivulja')
    plt.legend()
    plt.show()
#plot_roc_od_velikosti_ada()

############ gbrt ######################
def plot_AUC_od_ucenja_in_n_est():
    n = 10
    ucenje = np.linspace(0.1,2.0,10)
    n_est = np.arange(1,n+1, 1)
    (X,Y) = np.meshgrid(ucenje,n_est)
    AUC = np.zeros((len(ucenje), n))
    for indeks, vred in enumerate(ucenje):
        for j in range(1,n+1):
            gbrt = GradientBoostingClassifier(max_depth=1, n_estimators=j, learning_rate=vred)
            gbrt.fit(x_train, y_train)
            y_score = gbrt.predict_proba(x_val)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score)
            AUC[indeks][j-1] =  metrics.auc(fpr, tpr) ###ploščina
    c = plt.contourf(ucenje,n_est,AUC.T)
    b = plt.colorbar(c, orientation='vertical')
    b.set_label(r'$AUC$')
    plt.title(r'$globina=1$')
    plt.xlabel('hitrost učenja')
    plt.ylabel('število dreves')
    #plt.plot(fpr, tpr)
    plt.show()
    return None
#plot_AUC_od_ucenja_in_n_est_gbrt()

#plot_AUC_od_ucenja_in_n_est_gbrt()
def plot_AUC_od_max_gl_in_n_est_gbrt():
    m = 4
    n = 20
    globina = np.arange(1,m+1,1)
    n_est = np.arange(1,n+1, 1)
    (X,Y) = np.meshgrid(globina,n_est)
    AUC = np.zeros((m, n))
    for i in range(1,m+1):
        for j in range(1,n+1):
            gbrt = GradientBoostingClassifier(max_depth=i, n_estimators=j, learning_rate=0.1)
            gbrt.fit(x_train, y_train)
            y_score = gbrt.predict_proba(x_val)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score)
            AUC[i-1][j-1] =  metrics.auc(fpr, tpr) ###ploščina
    c = plt.contourf(globina,n_est,AUC.T)
    b = plt.colorbar(c, orientation='vertical')
    b.set_label(r'AUC')
    plt.xlabel('globina')
    plt.ylabel('število dreves')
    #plt.plot(fpr, tpr)
    plt.show()
    return None
#plot_AUC_od_max_gl_in_n_est_gbrt()
def plot_roc_od_globine_gbrt_s_predvidenim_st_dreves():
    globina = 5
    for i in range(1,globina+1):
        gbrt = GradientBoostingClassifier(max_depth=i, n_estimators=32)
        gbrt.fit(x_train, y_train)
        errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(x_val)]
        bst_n_estimators = np.argmin(errors)
        gbrt_best = GradientBoostingClassifier(max_depth=i,n_estimators=bst_n_estimators)
        gbrt_best.fit(x_train, y_train)
        y_score = gbrt_best.predict_proba(x_val)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score)
        plt.plot(fpr, tpr, label='{}, AUC= {:.2f}'.format(i, metrics.auc(fpr, tpr)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC krivulja')
    plt.legend()
    plt.show()
    return None
#plot_roc_od_globine_gbrt_s_predvidenim_st_dreves()
def napaka_od_stevila_dreves():
    drevesa = [i for i in range(1,201)]
    model = naredi_ada(3, 200, 1.0)
    model.fit(x_train, y_train)
    errors = [mean_squared_error(y_val, y_pred) for y_pred in model.staged_predict(x_val)]
    plt.plot(drevesa, errors)
    plt.plot(np.argmin(errors), np.min(errors), 'ro')
    plt.xlabel('število dreves')
    plt.ylabel('MSE')
    plt.show()

#napaka_od_stevila_dreves()



###############################################################
########## IMPLEMENTACIJA 'PLAYGORUND'ZGLEDOV #################
###############################################################
from data import spirale, kvadrati, mnozici, kroga

train_set, train_labels, test_set, test_labels = kroga()
#train_set, train_labels, test_set, test_labels = mnozici()
#train_set, train_labels, test_set, test_labels = kvadrati()
#train_set, train_labels, test_set, test_labels = spirale()
######mreža#######
Nx = 50
Ny = 50
grid_data = np.zeros((Nx*Ny, 2))
i = 0
for x in np.linspace(-2.0, 2.0, Nx):
	for y in np.linspace(-2.0, 2.0, Ny):
		grid_data[i] = [x, y]
		i += 1
#DNN:
model1 = naredi_model([32,32,32,], 2, True)
#history_data=model1[0].fit(epochs=30, batch_size=32,
#                x=train_set, y=train_labels,
#                validation_data=(test_set,test_labels), callbacks=[model1[1],])
#y_pred = model1[0].predict(test_set)
#grid_pred = model1[0].predict(grid_data)
#y_pred_bin = np.zeros_like(y_pred)
#y_pred_bin = y_pred > 0.5
#print(y_pred_bin)
#ADA:
model = naredi_ada(5, 50, 0.1)
model.fit(train_set, train_labels)
y_pred = model.predict(test_set)
grid_pred = model.predict(grid_data)
################################




barve_ozadje =ListedColormap(['#fff566', '#26ff58'])
barve_test =ListedColormap(['#edd81a', '#27a143'])

grid_pred_bin = np.zeros_like(grid_pred)
grid_pred_bin = grid_pred > 0.5
plt.scatter(grid_data[:, 0], grid_data[:, 1], c=grid_pred.reshape(-1), cmap= barve_ozadje)
plt.scatter(test_set[:, 0], test_set[:, 1], c=test_labels.reshape(-1), cmap=barve_test)
#plt.xlim(-2.5, 2.5)
#plt.ylim(-2.5, 2.5)
plt.title('globina:5, 50 dreves')
#plt.savefig('slike/ROC_slika.png')
plt.show()
