import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import pdb

df= pd.read_csv('datasets/DataETL.csv')
#X actual_polymer_weight1 must be non negative
dfFilter = df[(df['actual_polymer_weight1'] >= 0)] #& (df['recipe'] == 'XMT3317_105')

X= dfFilter.iloc[:,3:38]   #all features
Y= dfFilter.iloc[:,39]   #target output

#Select the top 3 features
best_features= SelectKBest(score_func=f_regression, k=10)
fit= best_features.fit(X,Y)

#create data frames for the features and the score of each feature
df_scores= pd.DataFrame(fit.scores_)
df_columns= pd.DataFrame(X.columns)



#combine all the features and their corresponding scores in one data frame
features_scores= pd.concat([df_columns, df_scores], axis=1)
features_scores.columns= ['Features', 'Score']
score = features_scores.sort_values(by = 'Score')

#print(score)

#split the dataset into X and Y
X= df[[  'loading_time'
        , 'setuptime'
        , 'oil_weighing_time'
        , 'idle_time'
        , 'carbon_black_act_weight'
        , 'carbon_weighing_time'
        , 'nptpl'
        , 'discharge_door_dwell_time'
        , 'set_dump_energy'
        , 'ram_up_time'
        , 'set_polymer_weight_2'
        , 'carbon_black_set_weight'
        , 'actual_polymer_weight_2'
        , 'manualinterruption'
        , 'carbon_charging_time'
        , 'carbonchargingrate (kg/sec)'
        , 'batch_temperature'
        , 'set_dump_temperature'
        , 'drop_door_close_time'
        , 'processoil_dust_stopinkg'
        , 'drop_door_open_time'
        , 'actual_polymer_weight_3'
        , 'set_polymer_weight_3'
        , 'mix_energy'
        , 'actual_polymer_weight_4'
        , 'set_polymer_weight_4'
        , 'oil_charging_time'
        , 'oil_actual_weight'
        , 'oil_set_weight'
        , 'set_polymer_weight_1'
        , 'actual_polymer_weight1'
        , 'total_weight'
        , 'nptcs'
        , 'ram_down_time'
        , 'mixingcycletime']]  #all features

Y= df[['MIXER_EFF']]  #the target output
#split the dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.6,random_state=10)

model = LinearRegression()
results = model.fit(X_train, y_train)

predictions = model.predict(X_test)

#r squared, coefficient of determination
modelScore = model.score(X_test, y_test)
print(modelScore)

#mean squared error
mse = metrics.mean_squared_error(y_test, predictions)
print(mse)

# prediction
# y_pred = model.predict(X_test)
# print(f"predicted response:\n{y_pred}")
#pdb.set_trace()