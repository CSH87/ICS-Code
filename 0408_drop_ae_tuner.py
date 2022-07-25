import pandas as pd
import numpy as np
import tensorflow as tf
# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# set random state
random_state =0

# read data

from pathlib import Path

my_file = Path('./drop_dul_data.csv')

if my_file.exists():
    drop_dul_data = pd.read_csv(my_file)
else:
    read_data = pd.read_csv('electra_modbus.csv')
    read_data = read_data.drop(['Time'],axis=1)
    # drop duplicate
    drop_dul_data = read_data.drop_duplicates()
    
# drop_dul_data.to_csv('drop_dul_data.csv',index=False)


# drop MitM_unaltered
tmp_mitm=drop_dul_data[drop_dul_data['label'] == 'MITM_UNALTERED'].index

drop_dul_data = drop_dul_data.drop(drop_dul_data[drop_dul_data['label'] == 'MITM_UNALTERED'].index)

# modify error columns values
drop_dul_data.loc[drop_dul_data.error >1 , 'error'] = 1


# separate data and label

load_label = drop_dul_data.loc[:,'label']

load_data = drop_dul_data.drop(['label'],axis=1)


########## data preproccessing

from sklearn.preprocessing import LabelEncoder

label_en = LabelEncoder()

load_data = pd.get_dummies(load_data)

for i in load_data.columns:
    load_data[i] = label_en.fit_transform(load_data[i])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

load_data['fc'] = scaler.fit_transform(load_data.loc[:,'fc'].values.reshape(-1, 1))
load_data['address'] = scaler.fit_transform(load_data.loc[:,'address'].values.reshape(-1, 1))
load_data['data'] = scaler.fit_transform(load_data.loc[:,'data'].values.reshape(-1, 1))

feature_name = load_data.columns
load_data_visual = load_data
# sum(load_data.iloc[:,6])
# to numpy format
load_data = load_data.values


########## for label

label_en.fit(load_label)
load_label = label_en.transform(load_label)

##### split training 


from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=random_state)

for train_index, test_index in split.split(load_data,load_label):
    train_data,test_data = load_data[train_index],load_data[test_index]
    train_label,test_label = load_label[train_index],load_label[test_index]

train_label = label_en.inverse_transform(train_label)
test_label = label_en.inverse_transform(test_label)


### set up denoising ae



# train_data_noisy = train_data


# noise_factor = 0.9
# random_normal = np.random.normal(loc=0.0, scale=1.0, size=(train_data.shape[0],))
# noise_clip = np.clip(noise_factor * random_normal, 0. , 1.).astype('int64')



# train_data_noisy[:,1] = train_data_noisy[:,1] + noise_clip
# train_data_noisy[:,3] = train_data_noisy[:,3] + noise_clip
# train_data_noisy[:,4] = train_data_noisy[:,4] + noise_clip


# build model

input_shape = train_data.shape[1]

def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    
    ae_layers = hp.Int('ae_layers', min_value=1, max_value=6, step=1)
    ae_dropout = hp.Float('drop_out',min_value=0.1,max_value=0.9,step=0.01)
    ae_units = hp.Choice('ae_units',[64,128,256,512])
    ae_activation = hp.Choice('activation',['relu','selu'])
    ae_initializer = hp.Choice('initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal','lecun_uniform','lecun_normal'])
    
    for i in range(ae_layers):
        model.add(tf.keras.layers.Dense(units=ae_units/2**i,activation=ae_activation,kernel_initializer=ae_initializer))
        model.add(tf.keras.layers.Dropout(rate=ae_dropout))
        
    for i in range(ae_layers,-1,-1):
        model.add(tf.keras.layers.Dense(units=ae_units/2**i,activation=ae_activation,kernel_initializer=ae_initializer))
        model.add(tf.keras.layers.Dropout(rate=ae_dropout))
        
    
    model.add(tf.keras.layers.Dense(units=input_shape,activation=ae_activation,kernel_initializer=ae_initializer))
    
    optimizer = hp.Choice('optimizer', ['adam', 'sgd','RMSprop'])
    model.compile(optimizer=optimizer,loss='mse')
    
    # model.summary()
    return model

import kerastuner

tuner = kerastuner.tuners.Hyperband(
    hypermodel=build_model,
    objective=kerastuner.Objective('val_loss', 'min'),
    max_epochs=100,
    directory='.',
    project_name='dae_0412',
    overwrite=True)

tuner.search_space_summary()
     
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss',patience=20,restore_best_weights=True)   


tuner.search(x=train_data,
             y=train_data,
             epochs=1000,
             batch_size=32,
             # callbacks=[early_stopping],
             validation_data=(test_data,test_data),
             verbose=1)

tuner.results_summary(1)

best_hyper = tuner.get_best_hyperparameters()[0].values

best_model = tuner.get_best_models()[0]

best_model.evaluate(train_data,train_data)

best_model.summary()

tf.keras.utils.plot_model(best_model, show_shapes=True)

# best_model.save('dropout_ae_0413.h5')
