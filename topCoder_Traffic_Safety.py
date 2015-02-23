from astropy.vo.samp.utils import _ServerProxyPoolMethod

__author__ = 'Michael'
import sklearn
import pandas as pd
import numpy as np
import sklearn_pandas
names = ["Id","source","dist","cycles",
         "complexity", "cargo","stops","start_day","start_month",
         "start_day_of_month","start_day_of_week",
         "start_time","days","pilot","pilot2",
         "pilot_exp","pilot_visits_prev","pilot_hours_prev",
         "pilot_duty_hrs_prev","pilot_dist_prev","route_risk_1",
         "route_risk_2","weather","visibility","traf0","traf1",
         "traf2","traf3","traf4","accel_cnt","decel_cnt","speed_cnt",
         "stability_cnt","evt_cnt"]
df = pd.DataFrame(data = pd.read_csv("../../Downloads/exampleData.csv", names = names))
print df.dtypes
df.start_time = pd.to_datetime(pd.Series(df.start_time)).convert_objects(convert_dates=True)

df.start_time = (np.array(df.start_time) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
df['dangerous'] = 0
df.ix[df['evt_cnt']>1, df['dangerous']] = 1
print type(df)
import sklearn.feature_extraction, sklearn.decomposition, sklearn.preprocessing
mapper = sklearn_pandas.DataFrameMapper([
(["evt_cnt","dangerous"], sklearn.preprocessing.OneHotEncoder())])

df['vectorized'] = 0

columns_predicting_with= ["Id","source","dist","cycles",
         "complexity", "cargo","stops","start_day","start_month",
         "start_day_of_month","start_day_of_week",
         "start_time","days","pilot","pilot2",
         "pilot_exp","pilot_visits_prev","pilot_hours_prev",
         "pilot_duty_hrs_prev","pilot_dist_prev","route_risk_1",
         "route_risk_2","weather","visibility","traf0","traf1",
         "traf2","traf3","traf4","accel_cnt","decel_cnt","speed_cnt",
         "stability_cnt"]
columns_predicting = ["evt_cnt"]

from sklearn import cross_validation
X_train, X_test, y_train, y_test = sklearn_pandas.cross_validation.train_test_split(df[columns_predicting_with],
                                                                                    df[columns_predicting],
                                                                                    test_size=0.33,
                                                                                    random_state=0)

from sklearn.metrics import accuracy_score
from sklearn.lda import LDA
clf = LDA()
clf.fit(X_train, y_train)
y_pred = []
#print clf.score(X_test, y_test)
correctPredictions = 0
bothCorrectPredictions = 0
totalPredictions = 0
for index, line in enumerate(X_test):
    pred = clf.predict(line)
    pred_evt_cnt = round(pred[0],0)
    #print pred[0]
    if pred_evt_cnt > 1:
        np.append(X_test[index], 1)
    else:
        np.append(X_test[index], 0)
    y_pred.append(round(pred[0],0))
    if (y_test[index] == pred_evt_cnt):
        correctPredictions +=1

        if(((X_test[index][-1] == 1 and
                 y_test[index] > 1)
        or (X_test[index][-1] == 0 and
                 y_test[index] <= 1 ))):

            bothCorrectPredictions += 1
    totalPredictions+=1

print accuracy_score(y_test,y_pred)
print sklearn.metrics.precision_score(y_test, y_pred)

print correctPredictions
print bothCorrectPredictions
print totalPredictions
