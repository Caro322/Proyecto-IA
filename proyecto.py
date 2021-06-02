import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Velocidad de onda de pulso

datos = pd.read_csv('signos.csv',sep=';', header=0) 
cabecera = ["Peso","Altura","Diastolica","Siastolica","Pulso","Temperatura","Muscular","Hidratacion","Huesos","VOP","Saturacion"] 
datos.columns = cabecera 
datos.head()

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

x1 = datos.values[:,0]
x2 = datos.values[:,1]
x3 = datos.values[:,2]
x4 = datos.values[:,3]
x5 = datos.values[:,4]
x6 = datos.values[:,5]
x7 = datos.values[:,6]
x8 = datos.values[:,7]
x9 = datos.values[:,8]
y = datos.values[:,9]
x10 = datos.values[:,10]

x0 = np.ones(x1.shape)

X = np.matrix([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]).T
Y = np.matrix ([y]).T

type(X)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

pca = decomposition.PCA(n_components=3,whiten=True,svd_solver='arpack')
pca.fit(X)
X = pca.transform(X)

#Regresión utilizando la transformada

Theta=np.linalg.inv(X.T*X)*(X.T)*Y
print(Theta)
plt.plot(x1,y,'bo')
plt.plot(x1, Theta[0,0]+Theta[1,0]*x1+Theta[2,0]*x2+Theta[3,0]*x3+Theta[4,0]*x4+Theta[5,0]*x5+Theta[6,0]*x6+Theta[7,0]*x7+Theta[8,0]*x8+Theta[9,0]*x9+Theta[10,0]*x10)
plt.title('Final')
plt.show()
R=np.corrcoef((Theta[0,0]+Theta[1,0]*x1+Theta[2,0]*x2+Theta[3,0]*x3+Theta[4,0]*x4+Theta[5,0]*x5+Theta[6,0]*x6+Theta[7,0]*x7+Theta[8,0]*x8+Theta[9,0]*x9+Theta[10,0]*x10),y)
R2=R**2
print(R2[0,1])

#Regresión lineal
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

partition=15000

X_train = X[:partition]
X_test = X[partition:]

y_train = Y[:partition]
y_test = Y[partition:]

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print('Coeficientes:', regr.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))

#Regresión con máquina de soporte vectorial

from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score

X_train = X[:partition]
X_test = X[partition:]

y_train = Y[:partition]
y_test = Y[partition:]

msv = svm.SVR(kernel='linear')

msv.fit(X_train, y_train)

y_pred = msv.predict(X_test)

print('Coeficientes:', msv.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))
msv = svm.SVR(kernel='rbf')

msv.fit(X_train, y_train)

y_pred = msv.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))
msv = svm.SVR(kernel='poly',degree=2)

msv.fit(X_train, y_train)

y_pred = msv.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))
msv = svm.SVR(kernel='poly',degree=3)

msv.fit(X_train, y_train)

y_pred = msv.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))

#Regresión con redes neuronales

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train = X[:partition]
X_test = X[partition:]

y_train = Y[:partition]
y_test = Y[partition:]

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))

#Predicción de admisión a la universidad

datos = pd.read_csv('Admission_Predict.csv',sep=',', header=0) 
cabecera = ["Serial","GRE","TOEFL","UniRating","Proposito","Recomendacion","GPA","Experiencia","P_Admisión"] 
datos.columns = cabecera 
datos.head()

datos.drop(["Serial"],axis=1,inplace=True)
datos.head()

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

x1 = datos.values[:,0]
x2 = datos.values[:,1]
x3 = datos.values[:,2]
x4 = datos.values[:,3]
x5 = datos.values[:,4]
x6 = datos.values[:,5]
x7 = datos.values[:,6]
y = datos.values[:,7]

x0 = np.ones(x1.shape)

X = np.matrix([x0,x1,x2,x3,x4,x5,x6,x7]).T
Y = np.matrix ([y]).T

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

pca = decomposition.PCA(n_components=3,whiten=True,svd_solver='arpack')
pca.fit(X)
X = pca.transform(X)

#Regresión utilizando la transformada

Theta=np.linalg.inv(X.T*X)*(X.T)*Y
print(Theta)
plt.plot(x1,y,'bo')
plt.plot(x1, Theta[0,0]+Theta[1,0]*x1+Theta[2,0]*x2+Theta[3,0]*x3+Theta[4,0]*x4+Theta[5,0]*x5+Theta[6,0]*x6+Theta[7,0]*x7)
plt.title('Final')
plt.show()
R=np.corrcoef((Theta[0,0]+Theta[1,0]*x1+Theta[2,0]*x2+Theta[3,0]*x3+Theta[4,0]*x4+Theta[5,0]*x5+Theta[6,0]*x6+Theta[7,0]*x7),y)
R2=R**2
print(R2[0,1])

#Regresión lineal

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

partition=300

X_train = X[:partition]
X_test = X[partition:]

y_train = Y[:partition]
y_test = Y[partition:]

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print('Coeficientes:', regr.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))

#Regresión con máquinas de soporte vectorial

from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score

X_train = X[:partition]
X_test = X[partition:]

y_train = Y[:partition]
y_test = Y[partition:]

msv = svm.SVR(kernel='linear')

msv.fit(X_train, y_train)

y_pred = msv.predict(X_test)

print('Coeficientes:', msv.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))

msv = svm.SVR(kernel='rbf')

msv.fit(X_train, y_train)

y_pred = msv.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))

msv = svm.SVR(kernel='poly',degree=2)

msv.fit(X_train, y_train)

y_pred = msv.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))

msv = svm.SVR(kernel='poly',degree=3)

msv.fit(X_train, y_train)

y_pred = msv.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))

#Regresión con redes neuronales

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train = X[:partition]
X_test = X[partition:]

y_train = Y[:partition]
y_test = Y[partition:]

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

print('Coeficiente de determinación: %.2f'
      % r2_score(y_test, y_pred))
