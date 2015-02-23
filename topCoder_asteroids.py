from __future__ import print_function
__author__ = 'Michael'
import pandas as pd
import numpy as np
import sklearn
"""names = ["Detection Number", "Frame Number","Sexnum", "Time",
         "RA","DEC","X","Y","Magnitude","FWHM","Elong","Theta","RMSE","Deltamu",
         "Rejected"]
dataFrame = pd.DataFrame(data=np.zeros((0,len(names))), columns=names)
from pandas import read_table
import os
for root, dirs, files in os.walk("../../Downloads/offline_training/offline_training"):
    print(root)
    print(dirs)
    print(files)
    for file in files:
        if file.endswith(".det"):
            filepath = str(os.path.join(root, file))
            print (filepath)
            try:
                df = pd.read_table(filepath, delimiter=" ", names = names )
                #print (df)

                dataFrame = dataFrame.append(df, ignore_index= True)
            except: BaseException
print(dataFrame)
"""
#dataFrame.to_csv("C:\Users\Michael\PycharmProjects\TopCoder\offline_training_dataFrame.csv")
df_csv = pd.read_csv("C:\Users\Michael\PycharmProjects\TopCoder\offline_training_dataFrame.csv", index_col = 0)
#print (df_csv[:13])
from sklearn import cross_validation
columns_predicting_with = ["Detection Number", "Frame Number","Sexnum", "Time",
         "RA","DEC","X","Y","Magnitude","FWHM","Elong","Theta","RMSE","Deltamu"]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_csv[columns_predicting_with],
                                                                     df_csv["Rejected"],
                                                                     test_size=0.33,
                                                                     random_state=0)
print (len(X_train))
print (len(X_test))
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
print ("Classification accuracy when cross-validating Naive Bayes Implementation of Classification using all Columns is " + str((clf.score(X_test, y_test))*100) + "%")

print("Can we do better?")

from sklearn import neighbors
n_neighbors = 100
for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train, y_train)
    print ("Classification accuracy when cross-validating K-NN Implementation of Classification using weights = " +
           weights+ " and all Columns is " + str((clf.score(X_test, y_test))*100) + "%")





from sklearn.svm import SVC
svm = SVC(kernel = "linear" )
svm.fit(X_train, y_train)
print ("Classification accuracy when cross-validating SVM Implementation of Classification using all Columns is " + str((svm.score(X_test, y_test))*100) + "%")


