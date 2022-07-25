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


# load_ dae result

load_dae = tf.keras.models.load_model('dropout_ae_0413.h5')

train_data = load_dae.predict(train_data)
test_data = load_dae.predict(test_data)



#### use smote

from imblearn.combine import SMOTETomek,SMOTEENN
from collections import Counter

print(Counter(label_en.inverse_transform(load_label)))

print(Counter(train_label))


# smto = SMOTETomek(random_state=random_state,sampling_strategy={'READ_ATTACK':5000,'REPLAY_ATTACK':5000})
# smte = SMOTETomek(random_state=random_state,sampling_strategy={'READ_ATTACK':5000,'REPLAY_ATTACK':5000})

# train_data, train_label = smte.fit_resample(train_data, train_label)

from imblearn.over_sampling import ADASYN 

ada = ADASYN(random_state=random_state,sampling_strategy={'READ_ATTACK':5000,'REPLAY_ATTACK':5000})
train_data, train_label = ada.fit_resample(train_data, train_label)
print(Counter(train_label))



###############
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import make_scorer

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

def score_metirics (true_label, predict_label,average):
    print(average)
    print('==========')
    print('Accuracy Score: {:.4f}'.format(accuracy_score(true_label, predict_label)))
    print('Precision Score: {:.4f}'.format(precision_score(true_label, predict_label, average=average)))
    print('Recall Score: {:.4f}'.format(recall_score(true_label, predict_label, average=average)))
    
    print('F1 Score: {:.4f}'.format(f1_score(true_label, predict_label, average=average)))
    print('==========')


def custom_score(y_true,y_pred):
    recall_ = recall_score(y_true, y_pred,average='macro')
    f1_ = f1_score(y_true, y_pred,average='macro')
    return recall_
    # return f1_
########## model

from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import kerastuner
from sklearn.model_selection import StratifiedKFold

def build_model(hp):
    model = XGBClassifier(
        learning_rate = hp.Float('learning_rate',1e-2,1),
        n_estimators=hp.Int('n_estimators',100,1000,step=100),
        max_depth=hp.Int('max_depth',1,10),
        n_jobs = hp.Int("n_jobs",-1,-1),
        
        
        )
        
    return model

num_trails = 12

tuner = kerastuner.tuners.Sklearn(
    oracle=kerastuner.oracles.BayesianOptimization(
        objective=kerastuner.Objective('score', 'max'),
        max_trials=num_trails),
    hypermodel = build_model,
    scoring = make_scorer(custom_score),
    cv = StratifiedKFold(5),
    directory='.',
    project_name='XGBClassifier_0413_f1',
    overwrite=True)

tuner.search(train_data, train_label)

# aaa = tuner.get_best_hyperparameters(num_trials=num_trails)

# for i in range(len(aaa)):
#     model = tuner.hypermodel.build(aaa[i])
#     model.fit(train_data,train_label)
#     disp = plot_confusion_matrix(model, test_data, test_label,xticks_rotation='vertical',
#                                   cmap=plt.cm.Blues,
#                                   normalize=None)
#     disp.ax_.set_title(i)
#     plt.show()
#     score_metirics(test_label,model.predict(test_data),'macro')

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

hp_model = tuner.hypermodel.build(best_hp)

hp_model.fit(train_data,train_label)


disp = plot_confusion_matrix(hp_model, test_data, test_label,xticks_rotation='vertical',
                                  cmap=plt.cm.Blues,
                                  normalize=None)
disp.ax_.set_title('test')
plt.show()

score_metirics(test_label,hp_model.predict(test_data),'macro')
score_metirics(test_label,hp_model.predict(test_data),'micro')

print(hp_model.set_params)


import joblib

# joblib.dump(hp_model,'./0423_hp_model.sav')




