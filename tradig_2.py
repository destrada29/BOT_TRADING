

#librerias selenium,time,numpy,sklearn

from fileinput import filename
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import numpy as np
import sklearn


#se crea un vector unidimecional para guardar los precios
n=np.zeros([100])
l=0

#como usamos selenuim para el web scraping le damos el directirio de donde esta el .exe
driver = webdriver.Chrome(executable_path=r"C:\Users\david\Downloads\chromedriver_win32\chromedriver.exe")

#empezamos la ejecucion y le damos la url de la cual vamos a sacar informacion 
validate=True
driver.get("https://www.binance.com/en/trade/BTC_BUSD?theme=dark&type=spot")   
time.sleep(1)

#aca buscamos en la pagina el dato o el div requerido y copiamos el xpath para extraer los valores
buscar_xpath=driver.find_element(By.XPATH,"//*[@id='spotOrderbook']/div[3]/div[2]/div[1]").text
time.sleep(1)

#este es un bucle que hace que cada vez que varia los datos del btc se obtengan hasta llegar al tope del vector (100)
while(validate==True):
    buscar_xpath=driver.find_element(By.XPATH,"//*[@id='spotOrderbook']/div[3]/div[2]/div[1]").text
    buscar_int=buscar_xpath.replace(",","")
    buscar_int=float(buscar_int)
    if(l<100):
        n[l]=buscar_int
        l=l+1
        print(n)
    else:
        validate=False

    time.sleep(1)


#cerramos la conexion a la pagina y trasponemos el vector para que quede de mas reciente a mas antiguo
driver.close()
n=n[::-1]
print(n)


#creamos la matriz de la cual vamos a hacerle la prediccion de la manera 1[23] 2[3,4]...
def matriz(n):
    comienzo=1
    necesaria=1
    fila=80
    x=0
    columna=21
    matriz=np.zeros([fila,columna])
    for o in range(fila):
        for p in range(columna):
            if(x<20 and comienzo<100):
                matriz[o,p]=n[comienzo]
                comienzo=comienzo+1
                x=x+1
            else:
                comienzo=necesaria+1

        necesaria=comienzo
        x=0 
    return(matriz)


matriz_resultado=matriz(n)

#generamos el output el cual se hace si el numero anterior al del principio del array es mayor,menor o igual
def output(matriz_resultado,n):

    for o in range (100):
        if(o<80):
            if(n[o]>matriz_resultado[o,0]):
                matriz_resultado[o,20]=float(10000.0)

            if(n[o]<matriz_resultado[o,0]):
                matriz_resultado[o,20]=float(20000.0)
            
            if(n[o]==matriz_resultado[o,0]):
                matriz_resultado[o,20]=float(30000.0)

        
    return(matriz_resultado)
       
    


matriz_resultado=output(matriz_resultado,n)
print(" ")
print(" ")
print(" ")
print(" ")

print(matriz_resultado[0:10,0:21])


#generamos el X y el Y  sabiendo que el X son los datos y el Y es el output
X = matriz_resultado[:,0:20]
print(" ")
print(" ")
print(X)
Y = matriz_resultado[:,20]
print(" ")
print(" ")
print(Y)


#partimos los datos con porcentaje de 80% training y 20% test
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=14541)


#lo que hacemos aca es escalar los datos para que quede mejor la presicion
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#importamos todas las librerias de cada una de las maneras de agrupar datos 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#ejecutamos cada una de las agrupaciones e imprimimos las presiciones 
Modelo_0 = KNeighborsClassifier(3)
Modelo_0.fit(X_train, Y_train)
Y_pred_0 =Modelo_0.predict (X_test)
print("Accuracy KNN",accuracy_score(Y_test, Y_pred_0))

Modelo_1 = GaussianNB()
Modelo_1.fit(X_train, Y_train)
Y_pred =Modelo_1.predict (X_test)
print("Accuracy Bayes",accuracy_score(Y_test, Y_pred))

Modelo_2 = LinearDiscriminantAnalysis()
Modelo_2.fit(X_train, Y_train)
Y_pred_2 =Modelo_2.predict (X_test)
print("Accuracy LDA",accuracy_score(Y_test, Y_pred_2))

Modelo_3 = QuadraticDiscriminantAnalysis()
Modelo_3.fit(X_train, Y_train)
Y_pred_3 =Modelo_3.predict (X_test)
print("Accuracy QDA",accuracy_score(Y_test, Y_pred_3))

Modelo_4 = DecisionTreeClassifier()
Modelo_4.fit(X_train, Y_train)
Y_pred_4 =Modelo_4.predict (X_test)
print("Accuracy Tree",accuracy_score(Y_test, Y_pred_4))

Modelo_5 = SVC()
Modelo_5.fit(X_train, Y_train)
Y_pred_5 =Modelo_5.predict (X_test)
print("Accuracy SVM",accuracy_score(Y_test, Y_pred_5))


#volvemos a abrir otro driver para recolectar los dato de una sola fila para comprobar que resultados nos da las agrupaciones dependeindo de testing
driver_1 = webdriver.Chrome(executable_path=r"C:\Users\david\Downloads\chromedriver_win32\chromedriver.exe")
validate=True
driver_1.get("https://www.binance.com/en/trade/BTC_BUSD?theme=dark&type=spot")   
buscar_xpath_1=driver_1.find_element(By.XPATH,"//*[@id='spotOrderbook']/div[3]/div[2]/div[1]").text
time.sleep(1)


l=0
testeo=np.zeros([1,20])
while(validate==True):
    buscar_xpath_1=driver_1.find_element(By.XPATH,"//*[@id='spotOrderbook']/div[3]/div[2]/div[1]").text
    buscar_int=buscar_xpath_1.replace(",","")
    buscar_int=float(buscar_int)
    testeo[:,l]=buscar_int
    print(testeo)
    l+=1
    if(l==20):
        validate=False
    time.sleep(1)


#le mandamos el array resultante y lo testeamos con cada metodo

testeo=scaler.transform(testeo)
Prediction_0 =Modelo_0.predict (testeo)
Prediction_1 =Modelo_1.predict (testeo)
Prediction_2 =Modelo_2.predict (testeo)
Prediction_3 =Modelo_3.predict (testeo)
Prediction_4 =Modelo_4.predict (testeo)
Prediction_5 =Modelo_5.predict (testeo)
print("La predicción de KNN es:",Prediction_0)
print("La predicción de Bayes es:",Prediction_1)
print("La predicción de LDA es:",Prediction_2)
print("La predicción de QDA es:",Prediction_3)
print("La predicción de Tree es:",Prediction_4)
print("La predicción de SVM es:",Prediction_5)

driver_1.close()







