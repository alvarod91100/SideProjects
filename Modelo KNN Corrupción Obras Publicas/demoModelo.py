import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt



#PREPARACION DE DATOS borrando toda fila con NaN
data= pd.read_csv("DatosContratos_4.csv")
data= data.loc[:, ["Tipo de procedimiento", "Estratificación de la empresa",'Importe del contrato', 'Periodo de Aprobación', 'RFC Fantasma', 'Supera Promedio para su Estratificación', "banderasPrueba"]]
data= data.dropna()

#Se separan variables dependientes e independientes
y= data.loc[:, "banderasPrueba"]
X= data.loc[:, ["Tipo de procedimiento", "Estratificación de la empresa",'Importe del contrato', 'Periodo de Aprobación', 'RFC Fantasma', 'Supera Promedio para su Estratificación']]

#Asignamos valores numericos a aquellos que son categóricos
dict = {'Adjudicación Directa Federal':0, 'Adjudicación directa':1, 'Invitación a cuando menos 3 personas': 2,  'Invitación a Cuando Menos 3 Personas':3 ,  'Licitación Pública': 4, 'Otro': 5,  'Proyecto de Convocatoria': 6}
X['Tipo de procedimiento'] = X['Tipo de procedimiento'].map(dict)
dict2= {'No MIPYME':0, 'Pequeña':1, 'Mediana': 2,  'Micro':3 ,  'nan': 4}
X['Estratificación de la empresa'] = X['Estratificación de la empresa'].map(dict2)


#Dividimos en train y test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)



#ESCALAR DATOS
scaler= RobustScaler()
X_test= scaler.fit_transform(X_test)
X_train= scaler.fit_transform(X_train)


#ENTRENAMIENTO
n=20
m=None
scores=[]

modelo= RandomForestClassifier(n_estimators=n, max_depth=m)
modelo.fit(X_train, y_train)
score=modelo.score(X_test, y_test)
scores.append(score)

for i in range(125, 150):
    modelo2= KNeighborsClassifier(n_neighbors=i)
    modelo2.fit(X_train, y_train)
    scores.append(modelo2.score(X_test, y_test))
    
    
rangeK= range(0, len(scores))
plt.figure()
plt.xlabel("K value")
plt.ylabel("Accuracy (%)")
plt.scatter(rangeK, scores)

optimalK= range(126, 155)[scores.index(max(scores))]




#MODELO CON MEJOR PRECISION
modelo2= KNeighborsClassifier(n_neighbors=optimalK)
modelo2.fit(X_train, y_train)
print("MODELO CON MAYOR PORCENTAJE:")
print(modelo2.score(X_test, y_test))





#Exportar resultados a CSV
Y_csv= modelo2.predict(X)
Y_csv=Y_csv.tolist()

X["Bandera Predicha"] = Y_csv
resultados_csv= X
resultados_csv.to_csv("Resultados.csv")

