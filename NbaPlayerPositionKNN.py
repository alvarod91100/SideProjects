
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


#Consider featue selection as well



scaler= RobustScaler()
Players= pd.read_csv("NBA_Players.csv")
#print(Players)
features= [[' HT', ' WT', ' BLKPG', ' FTP', ' RGP_CAREER']]



for feat in features: 
    Players[feat]= scaler.fit_transform(Players[feat])
    

    
Data= Players[[' POSITION', ' HT', ' WT', ' BLKPG', ' FTP', ' RGP_CAREER']]
X= Data[[' HT', ' WT', ' BLKPG', ' FTP', ' RGP_CAREER']]
y= Data[' POSITION']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)


from sklearn.neighbors import KNeighborsClassifier
#creas el objeto del Modelo
classifier= KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
firstAcc=classifier.score(X_test, y_test)
print(firstAcc)
PlayerPositionPrediction = classifier.predict(X_test)

#print(y_test)
#print(PlayerPositionPrediction)



#ERROR BY K
rangeK= range(1,20)
accuracy=[]
for k in rangeK:
    ClassifierK= KNeighborsClassifier(n_neighbors=k)
    ClassifierK.fit(X_train,y_train)
    accuracy.append(ClassifierK.score(X_test, y_test))

plt.figure()
plt.xlabel("K value")
plt.ylabel("Accuracy (%)")
plt.scatter(rangeK, accuracy)

optimalK= rangeK[accuracy.index(max(accuracy))]



classifier= KNeighborsClassifier(n_neighbors=optimalK)
classifier.fit(X_train, y_train)
predictionsNew= classifier.predict(X_test)
print('Accuracy of model with k=3: ' + str(firstAcc) + "\n" + 'Accuracy of model with k=' + str(optimalK) + ": " + str(classifier.score(X_test, y_test)))


Y_pred= pd.Series(predictionsNew)
Y_pred= pd.DataFrame(Y_pred)

dfComparison = pd.concat([y_test, Y_pred], axis=0)
print(Y_pred)
print(y_test)
