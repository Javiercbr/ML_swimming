import pandas as pd
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns
 

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def times2seconds(data_frame):

    '''converts a strin MM:SS to seconds'''
    
    aux = data_frame["alt_adj_swim_time_formatted"] 
    
    for i in range(len(aux)):
        
        if aux.loc[i].find(':')==1:
            
            b = aux.loc[i].split(':')
            b0 = float(b[0]) 
            b1 = float(b[1]) 
            c = str(60.0*b0 + b1)
            aux.loc[i]=c

 
    data_frame["alt_adj_swim_time_formatted"] = aux
    return data_frame



##############################################################################################################
##############################################################################################################
##############################################################################################################
###################################### performance of 18 yo swimmers #########################################

# opening csv files
data18  = pd.read_csv("SWrecordsF\Report_AAAA_18_F.csv")
data18  = data18.append(pd.read_csv("SWrecordsF\Report_AAA_18_F.csv"))
data18  = data18.append(pd.read_csv("SWrecordsF\Report_AA_18_F.csv"))
data18  = data18.append(pd.read_csv("SWrecordsF\Report_A_18_F.csv"))
data18  = data18.append(pd.read_csv("SWrecordsF\Report_BB_18_F.csv")) 


# removing problematic characters
data18 = data18.replace('=','', regex=True)
data18 = data18.replace('"','', regex=True)
data18.columns = data18.columns.str.replace('=', '')
data18.columns = data18.columns.str.replace('"', '')
 
# removing duplicates for the same record
data18 = data18.drop_duplicates(subset=['time_id'])
 

# getting swimmer names
swimmer_name = data18['full_name'].drop_duplicates()


data = { 'name':[],  't_medio': [], 't_max': [], 't_min':[], 't_std':[] }
records18 = pd.DataFrame(data)

for i in  swimmer_name:
  
    aux = data18.loc[data18['full_name'] == i ]
     
    media   =  (aux["alt_adj_swim_time_formatted"].astype('float')).mean()
    maxx    =  (aux["alt_adj_swim_time_formatted"].astype('float')).max()
    minn    =  (aux["alt_adj_swim_time_formatted"].astype('float')).min()
    desvio  =  (aux["alt_adj_swim_time_formatted"].astype('float')).std(ddof=0)  #std()
 
    fila =[i, media, maxx, minn, desvio]
    records18.loc[len(records18)] = fila
records18.head()  

print("There are ", len(records18), "unique records of 18 yo swimmers." )

##############################################################################################################
##############################################################################################################
##############################################################################################################



##############################################################################################################
##############################################################################################################
##############################################################################################################
###################################### performance of older swimmers #########################################


 
data_young = pd.read_csv("SWrecordsF\Report_AAAA_15_F.csv")
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAAA_14_F.csv"))
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAAA_13_F.csv"))
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAAA_12_F.csv"))
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAAA_11_F.csv"))
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAAA_10_F.csv"))                             
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAA_15_F.csv") )
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAA_14_F.csv") )
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAA_13_F.csv") )
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAA_12_F.csv") )
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAA_11_F.csv") )
data_young = data_young.append(pd.read_csv("SWrecordsF\Report_AAA_10_F.csv") )                            
data_young.head() 
 


# removing undesired characters
data_young = data_young.replace('=','', regex=True)
data_young=data_young.replace('"','', regex=True)
data_young.columns = data_young.columns.str.replace('=', '')
data_young.columns = data_young.columns.str.replace('"', '')
data_young.head()

   
data_young = data_young.drop_duplicates(subset=['time_id'])
data_young = data_young.reset_index()

data_young = times2seconds(data_young) # converts time to seconds

young_swimmer_name = data_young['full_name'].drop_duplicates()
 
print("There are ", len(young_swimmer_name), " unique records of 18 young swimmers." )

data = { 'name':[],  't_medio10': [], 't_max10': [], 't_min10':[], 't_std10':[],
                     't_medio11': [], 't_max11': [], 't_min11':[], 't_std11':[],
                     't_medio12': [], 't_max12': [], 't_min12':[], 't_std12':[], 
                     't_medio13': [], 't_max13': [], 't_min13':[], 't_std13':[], 
                     't_medio14': [], 't_max14': [], 't_min14':[], 't_std14':[], 
                     't_medio15': [], 't_max15': [], 't_min15':[], 't_std15':[]   }    
       
       

records_young = pd.DataFrame(data)
fila = np.zeros((6,4))

edades = [ '10', '11', '12', '13', '14' ,'15' ]
for i in  young_swimmer_name:
    
    aux0 = data_young.loc[data_young['full_name'] == i  ]
    
    for j in  range(len(edades)):
        aux = aux0.loc[aux0['swimmer_age'] == edades[j] ]
     
        media   =  (aux["alt_adj_swim_time_formatted"].astype('float')).mean()
        maxx    =  (aux["alt_adj_swim_time_formatted"].astype('float')).max()
        minn    =  (aux["alt_adj_swim_time_formatted"].astype('float')).min()
        desvio  =  (aux["alt_adj_swim_time_formatted"].astype('float')).std(ddof=0) #.std()
    
        fila[j] =[media, maxx, minn, desvio]
    
    records_young.loc[len(records_young)] = [i] + fila[0].tolist()   + fila[1].tolist()   + fila[2].tolist()   + fila[3].tolist()   + fila[4].tolist()   + fila[5].tolist()   
 
    
records_young.head() 




young_swimmers = set(records_young['name'].values)
old_swimmers   = set(    records18['name'].values)
swimmers_in_both_groups = young_swimmers.intersection(old_swimmers)

print("There are ", len(swimmers_in_both_groups), "swimmers in both groups.")

records_young_new = records_young.loc[records_young['name'].isin(list(swimmers_in_both_groups))] 
records18_new = records18.loc[records18['name'].isin(list(swimmers_in_both_groups))] 



# check for incomplete records
r = records_young_new [  ['t_medio10','t_medio11','t_medio12','t_medio13','t_medio14','t_medio15' ] ]
aa = np.isnan(  r ).astype(int)
zz = aa.sum(axis = 1)
plt.hist(zz, 20)
plt.xlabel('Number of missing records')
plt.show()

records_young_new['cant']  = zz # number of ages with missing records


records_young_new_2 = records_young_new.loc[records_young_new['cant'] == 0 ]
final_swimmers = records_young_new_2['name'].values 
records18_new_2 = records18_new.loc[records18_new['name'].isin(list(final_swimmers))] 
records18_new_2 = records18_new_2.reset_index(drop=True)
records_young_new_2 = records_young_new_2.reset_index(drop=True)


records_young_new_3 = records_young_new.loc[ (records_young_new['cant'] == 0) | (records_young_new['cant'] == 1) | (records_young_new['cant'] == 2) ] # | (records_young_new['cant'] == 3)  ]
print(len(records_young_new_3))
final_swimmers = records_young_new_3['name'].values 
records18_new_3 = records18_new.loc[records18_new['name'].isin(list(final_swimmers))] 
print(len(records18_new_3))
records18_new_3 = records18_new_3.reset_index(drop=True)
records_young_new_3 = records_young_new_3.reset_index(drop=True)
print(records_young_new_3['name'].equals(records18_new_3['name']))


final_data = records_young_new_2.set_index('name').join(records18_new_2.set_index('name'))
final_data = final_data.drop( ['cant'], axis = 1 )

 
# to fill missing records with median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
 
print(len(final_data))    
aux = imputer.fit_transform(final_data)
final_data = pd.DataFrame(aux, columns= final_data.columns)


# substitutes std = 0 (single times) by median. This is because a single record may suggest a very stable swimmer
final_data['t_std10'].loc[final_data['t_std10']==0]=final_data['t_std10'].median()
final_data['t_std11'].loc[final_data['t_std11']==0]=final_data['t_std11'].median()
final_data['t_std12'].loc[final_data['t_std12']==0]=final_data['t_std12'].median()
final_data['t_std13'].loc[final_data['t_std13']==0]=final_data['t_std13'].median()
final_data['t_std14'].loc[final_data['t_std14']==0]=final_data['t_std14'].median()
final_data['t_std15'].loc[final_data['t_std15']==0]=final_data['t_std15'].median()
final_data['t_std'].loc[final_data['t_std']==0]=final_data['t_std'].median()



# scaling times to [0, 1]
from sklearn import preprocessing
x = final_data.drop(['t_medio', 't_max','t_min','t_std'], axis = 1).values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
final_data[ ['t_medio10', 't_max10', 't_min10', 't_std10', 't_medio11', 't_max11', 't_min11', 't_std11',
           't_medio12', 't_max12', 't_min12', 't_std12', 't_medio13', 't_max13', 't_min13', 't_std13', 
           't_medio14', 't_max14', 't_min14', 't_std14', 't_medio15', 't_max15', 't_min15', 't_std15']] =  x_scaled



# splitting data
train_df, test_df = train_test_split(final_data, test_size=0.25, random_state=42)

 
# removing target
train_num_scaled = train_df.drop(['t_medio', 't_max','t_min','t_std'], axis = 1)
test_num_scaled  = test_df.drop( ['t_medio', 't_max','t_min','t_std'], axis = 1) 


X_train = train_num_scaled
y_train = train_df['t_medio']
 
    
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV    
# simple Supprot Vector Regressor 
modelo_SVR=SVR() 


hiperparametros_SVR =  {'C': [0.1, 0.5, 1, 2 ], 'epsilon': [0.1, 1, 10], 'degree': [3, 4, 5, 6, 8, 11, 15 ],'kernel':['linear', 'rbf'] }

# best parameters using crossvalidation 
grilla = GridSearchCV(modelo_SVR, param_grid = hiperparametros_SVR, cv=5, refit=True, scoring = 'neg_mean_absolute_error')
grilla.fit(X_train,y_train)


SVR_optimo = grilla.best_estimator_          



std_SVR   =  grilla.cv_results_['mean_test_score'].std()
media_SVR = -grilla.cv_results_['mean_test_score'].mean()
print(media_SVR)
print(std_SVR)

