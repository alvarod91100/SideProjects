import random as ran
import pandas as pd
import numpy as np
import scipy 


datosContratos= pd.read_csv("Contratos2018.csv", encoding = "ISO-8859-1")
#conservar = pd.isnull(datosContratos["Estratificación de la empresa"])
#datosContratos = datosContratos.loc[~conservar]
rfc= pd.read_csv("Definitivos.csv", encoding = "ISO-8859-1", skiprows=2)
rfc=rfc["RFC"]
rfc=rfc.tolist()



dicc = pd.read_excel ("CompraNet-Contratos-diccionario.xlsx")
dicc= dicc.iloc[5:,:]

datosContratos['Fecha de apertura'] = pd.to_datetime(datosContratos['Fecha de apertura'])
datosContratos['Fecha de fallo'] = pd.to_datetime(datosContratos['Fecha de fallo'])
datosContratos['Fecha de publicación'] = pd.to_datetime(datosContratos['Fecha de publicación'])
datosContratos['Fecha de inicio del contrato'] = pd.to_datetime(datosContratos['Fecha de inicio del contrato'])
datosContratos['Fecha de fin del contrato'] = pd.to_datetime(datosContratos['Fecha de fin del contrato'])
datosContratos['Fecha de firma del contrato'] = pd.to_datetime(datosContratos['Fecha de firma del contrato'])
#datosContratos["Fecha de firma del contrato"] = datosContratos["Fecha de firma del contrato"].dt.tz_localize(None)
datosContratos['Periodo de Aprobación']= datosContratos['Fecha de inicio del contrato'] - datosContratos['Fecha de publicación']

for i in range(0, len(datosContratos["Periodo de Aprobación"])):
    datosContratos.loc[i, "Periodo de Aprobación"] = datosContratos.loc[i, "Periodo de Aprobación"].days

#SEGUIR VIENDO LO DE LAS FECHAS
#--------------------------------------------------------------------------------------------------------------------------------------


#ESTRATIFICACION
estrat= 'Estratificación de la empresa'
importe= "Importe del contrato"
tipos = datosContratos['Estratificación de la empresa'].unique()


NoMIPYME_Bool= datosContratos.loc[:,estrat] == tipos[0]
NoMIPYME_Importe= datosContratos[importe][NoMIPYME_Bool]
tamNoPYME=len(NoMIPYME_Importe)
promNOMIPYME= sum(NoMIPYME_Importe)/tamNoPYME

peq_Bool= datosContratos.loc[:,estrat] == tipos[1]
peq_Importe= datosContratos[importe][peq_Bool]
tam_Peq=len(peq_Importe)
promPeq= sum(peq_Importe)/tam_Peq

med_Bool= datosContratos.loc[:,estrat] == tipos[3]
med_Importe= datosContratos[importe][med_Bool]
med_Tam=len(med_Importe)
prom_Med= sum(med_Importe)/med_Tam

micro_Bool= datosContratos.loc[:,estrat] == tipos[4]
micro_Importe= datosContratos[importe][micro_Bool]
micro_Tam=len(micro_Importe)
prom_Micro= sum(micro_Importe)/micro_Tam


nombreCol= "Supera Promedio para su Estratificación"
valores= []
for i in range(0, 194191):
    if datosContratos.loc[i, estrat] == "No MIPYME":
        valores.append(datosContratos.loc[i, importe]-promNOMIPYME)
 
    if datosContratos.loc[i, estrat] == "Pequeña":
      
        valores.append(datosContratos.loc[i, importe]-promPeq)
      
            
    if datosContratos.loc[i, estrat]== "Mediana":
     
        valores.append(datosContratos.loc[i, importe] - prom_Med)
     

    if datosContratos.loc[i, estrat]== "Micro":
        valores.append(datosContratos.loc[i, importe]-prom_Micro)
       
     
    if pd.isnull(datosContratos.loc[i, estrat]):
        valores.append(np.nan)
    
datosContratos[nombreCol]= valores






#---------------------------------------------------------------------------------------
#Comparacion con dataset de RFC Fantasmas
rfc_datos= datosContratos["RFC"]
rfc_datos.tolist()

RFC_fant=[]

for i in range(0, 194191):
    if rfc_datos[i] in rfc:
        RFC_fant.append(1)
    if rfc_datos[i] not in rfc:
        RFC_fant.append(0)



datosContratos["RFC Fantasma"]= RFC_fant 

#---------------------------------------------------------------------------------------------






#En caso de si haber habido una columna que nos permitiria entrenar las predicciones 
#del modelo, lo siguiente no 
#seria necesario, pero se esta simulando esto con probabilidades para simular el 
#tener una base de datos con esta columna que se necesita

#Esto se podria hacer en el futuro con gente experimentada en esta 
# clase de analisis para entrenar el modelo de nuevo una vez que se tengan
# los datos de la variable a predecir ya observadas y no simuladas
#-----------------------------------------------------------------------------------------------------------------------------------------

banderas= ["verde", "amarillo", "rojo"]
banderas_dataset=[]

for i in range(0, 194191):
    #if datosContratos.loc[i, "Tipo de procedimiento"]== "Adjudicación Directa Federal":
    if datosContratos.loc[i, "RFC Fantasma"] == 1:
        banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0, 0,1])])
        
        
    else:
        if datosContratos["Periodo de Aprobación"][i].days <= 4 and datosContratos["Periodo de Aprobación"][i].days > 0:
            banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[.6 ,0.25, 0.15])])
        
        elif datosContratos["Periodo de Aprobación"][i].days <= 0  and datosContratos["Periodo de Aprobación"][i].days > -7:
            banderas_dataset.append(banderas[np.random.choice(np.arange(1,3), p=[0.75, 0.25])])
           
        elif datosContratos["Periodo de Aprobación"][i].days <= -7:
            banderas_dataset.append(banderas[np.random.choice(np.arange(1,3), p=[0.65, 0.35])])
           
        else: 
            if datosContratos.loc[i, estrat] == "No MIPYME":  
                if datosContratos.loc[i, importe] > promNOMIPYME:
                    banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.7,0.2, 0.1])])
                    #if datosContratos.loc[i, importe] < 30000000:    
                        #   banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.75, 0.2, 0.05])])
                        #if datosContratos.loc[i, importe] > 30000000:    
                            #   banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.7, 0.15, 0.15])])
                            
                if datosContratos.loc[i, importe] <= promNOMIPYME:    
                    banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.7, 0.25, 0.05])])    

        
            if datosContratos.loc[i, estrat] == "Pequeña": 
                if datosContratos.loc[i, importe] > promPeq:
                    banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.6,0.3, 0.1])])
                #if datosContratos.loc[i, importe] < 100000000:
                #   banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.7, 0.2, 0.1])])
                #if datosContratos.loc[i, importe] > 100000000:
                #   banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.6, 0.3, 0.1])])
                if datosContratos.loc[i, importe] <= promPeq:    
                    banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.7, 0.25, 0.05])])    
        
    
        
            if datosContratos.loc[i, estrat] == "Mediana": 
                if datosContratos.loc[i, importe] > prom_Med:
                    banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.55, 0.35, 0.1])])
                    #if datosContratos.loc[i, importe] < 3000000000:
                        #   banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.65, 0.3, 0.05])])
                        #if datosContratos.loc[i, importe] > 3000000000:
                            #   banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.6, 0.3, 0.1])])
                if datosContratos.loc[i, importe] <= prom_Med:  
                    banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.7, 0.25, 0.05])])    
    
    
            if datosContratos.loc[i, estrat] == "Micro": 
                if datosContratos.loc[i, importe] > prom_Micro:
                    banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[.5,0.3, 0.2])])
                #if datosContratos.loc[i, importe] < 10000000:
                #   banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.6, 0.3, 0.1])])
             #if datosContratos.loc[i, importe] > 10000000:
                #   banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.55, 0.3, 0.15])])
                if datosContratos.loc[i, importe] <= prom_Micro:   
                    banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.7, 0.25, 0.05])])    
   
    
            if datosContratos.loc[i, estrat] != "No MIPYME":
                if datosContratos.loc[i, estrat] != "Pequeña":
                    if datosContratos.loc[i, estrat] != "Mediana":  
                        if datosContratos.loc[i, estrat] != "Micro": 
                            banderas_dataset.append(banderas[np.random.choice(np.arange(0,3), p=[0.85, 0.1, 0.05])])
            
        

datosContratos["banderasPrueba"]= banderas_dataset 












#------------------------------------------------------------------------------------------------------------------------
#DATOS DE INTERES
bool_RFCfant= datosContratos.loc[:,"RFC Fantasma"] == 1
Importe_RFCFant= datosContratos[bool_RFCfant]











    
    
    
    
    
    
    
    
    
    
    

