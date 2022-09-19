from pyexpat import model
from unittest import skip
from scipy.special import yv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp


# Uso de regresion lineal multiple para predecir el precio de los autos 


#Lectura CSV//dataset
data = pd.read_csv('./dataset/CarPrice_Assignment.csv')
print(data.head(5))
print("\n")

#Se puede observar que nuestra base tiene 26 variables siendo varias de estas categoricas por lo que tendremos que convertir esos valores a nuemeros, 
#para poder implementar el modelo de regresión lineal
#Pero antes de eso checaremos para saber si hay algun valor faltante en nuestra base

print('Cuenta de datos nulos: ')
print(data.isnull().sum())
print('No se cuenta con datos nulos')

#Despues revisaremos por outliers en nuestra variable dependiente para asi determinar el sesgo del modelo

sns.boxplot( x = data["price"])
plt.show()

# Se puede observar en nuestra grafica que visualmente el sesgo del precio del auto es muy alto dandonos un sesgo positivo
# Comprobaremos esto con la siguiente funcion

print(data.skew(axis = 0, skipna = True))

# efectivamente el precio nos da un sesgo positivo con un valor alto de 1.777678
# Ya que este sesgo es muy alto escalaremos los datos


data['CarName'] = data['CarName'].astype('category')
data['CarName'] = data['CarName'].cat.codes

data['fueltype'] = data['fueltype'].astype('category')
data['fueltype'] = data['fueltype'].cat.codes

data['aspiration'] = data['aspiration'].astype('category')
data['aspiration'] = data['aspiration'].cat.codes

data['doornumber'] = data['doornumber'].astype('category')
data['doornumber'] = data['doornumber'].cat.codes

data['carbody'] = data['carbody'].astype('category')
data['carbody'] = data['carbody'].cat.codes

data['drivewheel'] = data['drivewheel'].astype('category')
data['drivewheel'] = data['drivewheel'].cat.codes

data['enginelocation'] = data['enginelocation'].astype('category')
data['enginelocation'] = data['enginelocation'].cat.codes

data['enginetype'] = data['enginetype'].astype('category')
data['enginetype'] = data['enginetype'].cat.codes

data['cylindernumber'] = data['cylindernumber'].astype('category')
data['cylindernumber'] = data['cylindernumber'].cat.codes


data['fuelsystem'] = data['fuelsystem'].astype('category')
data['fuelsystem'] = data['fuelsystem'].cat.codes

print(data.head(5))

#Ya con los datos categoricos cambiados realizaremos una matriz de correlación para saber que datos modifican el precio

plt.figure(figsize= (12,8))
sns.heatmap(data.corr() , annot = True , cmap = "YlGnBu")
plt.show()

#En nuestra matriz de correlacion se observan varias correlaciones positivas, lo que nos dice que el precio sube dependiendo de la cantidad de estas variables,
#Es por esto que para la realización de nuestro modelo usaremos solo las que tienen mayor correlación, estas son:
# curbweight, enginesize, horsepower, carlenght, carwidth, aspiration, driver wheel, wheel base, cylindernumber, doornumber y carbody.
# crearemos una nueba base de datos solo con estas variables para realizar nuestro modelo.

dataModelo = data.copy()
dataModelo = dataModelo[['doornumber','carbody','drivewheel', 'wheelbase','carlength','carwidth', 'aspiration', 'cylindernumber', 'curbweight','enginesize','boreratio','horsepower','price']]

print(dataModelo.head(5))

#Con esta nueva base ya podemos empezar a separa nuestros datos de entrenamiento, testeo y validacion.


#Separación de variables dependientes e independientes


x =dataModelo.drop(["price"],axis=1).values
y= dataModelo['price'].values



#Separacion del dataset de entrenamiento
x_main,x_test,y_main,y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

#Separación de datos de validación
x_train,x_val,y_train,y_val = train_test_split(x_main, y_main, test_size= 0.2, random_state = 4)


#Como hemos dicho anteriormete nuestra media esta sesgadda por lo que escalaremos los datos al rango cuantilico

# ro_scaler = RobustScaler()
# x_train = ro_scaler.fit_transform(x_train)
# x_test = ro_scaler.fit_transform(x_test)
# x_val = ro_scaler.fit_transform(x_val)



#Ya con los datos escalados y separados crearemos nuestro modelo

modelo = linear_model.LinearRegression()

#Entrenamos nuestro modelo con los datos de entrenamiento
modelo.fit(x_train,y_train)

#Realizamos predicciones para las variables de testeo

predTest = modelo.predict(x_test)
prediccionesTest = pd.DataFrame({"precio_real_Test": y_test , "Prediccion" : predTest})

plt.figure(figsize=(20,12))
plt.plot(prediccionesTest[:100])
plt.legend(['Valores esperados' , 'Prediccion'])
plt.show()

#Realizamos predicciones para la validacion

predVal = modelo.predict(x_val)
prediccionesVal = pd.DataFrame({"precio_real_val": y_val , "Prediccion" : predVal})
plt.figure(figsize=(20,12))

plt.plot(prediccionesVal[:100])
plt.legend(['Valores esperados' , 'Prediccion'])
plt.show()

print(prediccionesVal.head(15))

#imprimimos el score en el tsteo y la validacion
print(modelo.score(x_test,y_test))
print(modelo.score(x_val,y_val))