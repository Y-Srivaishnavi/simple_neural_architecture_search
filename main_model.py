import pandas as pd
import pickle
import numpy as np
from randomNAS import randomsearch
from randomvariables import nas_data_log

# read the data
train_data = pd.read_csv('DATA/train.csv')
val_data = pd.read_csv('DATA/val.csv')

# split it into X and y values
x = np.array(train_data.drop(['label','filename','patient_id'], axis=1, inplace=False)).astype('float32')
#y = pd.get_dummies(data['label']).values
y = (train_data['label']).values

#validation dataset
x_val = np.array(val_data.drop(['label','filename','patient_id'], axis=1, inplace=False)).astype('float32')
y_val = (val_data['label']).values

# let the search begin
data = randomsearch(x,y,x_val,y_val)

#log data
with open(nas_data_log, 'wb') as f:
    pickle.dump(data, f)
