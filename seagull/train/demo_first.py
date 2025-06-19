# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:45:18 2024

@author: awei
"""

import warnings
warnings.filterwarnings('ignore')

import os, gc
import pandas as pd
import numpy as np
import janestreet
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from joblib import dump, load
import datatable as dtable

import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

TEST = False

def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2**(n + 1 - j)))
    return np.average(a, weights = w)
if TEST:
    train = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv', nrows = 100)
    features = [c for c in train.columns if 'feature' in c]
else:
    print('Loading...')
    train = dtable.fread('../input/jane-street-market-prediction/train.csv').to_pandas()
    features = [c for c in train.columns if 'feature' in c]

    print('Filling...')
    train = train.query('date > 85').reset_index(drop = True) 
    train = train.query('weight > 0').reset_index(drop = True)
    train[features] = train[features].fillna(method = 'ffill').fillna(0)
    train['action'] = ((train['resp_1'] > 0) & (train['resp_2'] > 0) & (train['resp_3'] > 0) & (train['resp_4'] > 0) & (train['resp'] > 0)).astype('int')

    resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']

    X = train[features].values
    y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
    date = train['date'].values
    weight = train['weight'].values
    resp = train['resp'].values
    sw = np.mean(np.abs(train[resp_cols].values), axis = 1)

n_splits = 5
group_gap = 31
def create_ae_mlp(num_columns, num_labels, hidden_units, dropout_rates, ls = 1e-2, lr = 1e-3):
    
    inp = tf.keras.layers.Input(shape = (num_columns, ))
    x0 = tf.keras.layers.BatchNormalization()(inp)
    
    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)
    
    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name = 'decoder')(decoder)

    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('swish')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)

    out_ae = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name = 'ae_action')(x_ae)
    
    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)
    
    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)
        
    out = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name = 'action')(x)
    
    model = tf.keras.models.Model(inputs = inp, outputs = [decoder, out_ae, out])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                  loss = {'decoder': tf.keras.losses.MeanSquaredError(), 
                          'ae_action': tf.keras.losses.BinaryCrossentropy(label_smoothing = ls),
                          'action': tf.keras.losses.BinaryCrossentropy(label_smoothing = ls), 
                         },
                  metrics = {'decoder': tf.keras.metrics.MeanAbsoluteError(name = 'MAE'), 
                             'ae_action': tf.keras.metrics.AUC(name = 'AUC'), 
                             'action': tf.keras.metrics.AUC(name = 'AUC'), 
                            }, 
                 )
    
    return model
params = {'num_columns': len(features), 
          'num_labels': 5, 
          'hidden_units': [96, 96, 896, 448, 448, 256], 
          'dropout_rates': [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448], 
          'ls': 0, 
          'lr':1e-3, 
         }
if not TEST:
    scores = []
    batch_size = 4096
    gkf = PurgedGroupTimeSeriesSplit(n_splits = n_splits, group_gap = group_gap)
    for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
        ckp_path = f'JSModel_{fold}.hdf5'
        model = create_ae_mlp(**params)
        ckp = ModelCheckpoint(ckp_path, monitor = 'val_action_AUC', verbose = 0, 
                              save_best_only = True, save_weights_only = True, mode = 'max')
        es = EarlyStopping(monitor = 'val_action_AUC', min_delta = 1e-4, patience = 10, mode = 'max', 
                           baseline = None, restore_best_weights = True, verbose = 0)
        history = model.fit(X[tr], [X[tr], y[tr], y[tr]], validation_data = (X[te], [X[te], y[te], y[te]]), 
                            sample_weight = sw[tr], 
                            epochs = 100, batch_size = batch_size, callbacks = [ckp, es], verbose = 0)
        hist = pd.DataFrame(history.history)
        score = hist['val_action_AUC'].max()
        print(f'Fold {fold} ROC AUC:\t', score)
        scores.append(score)

        K.clear_session()
        del model
        rubbish = gc.collect()
    
    print('Weighted Average CV Score:', weighted_average(scores))
    
    
Training details:
In this competition, I have used a supervised autoencoder MLP approach and my teammates use XGBoost. Our final submission is a simple blend of these two models. Here, I would like to explain my approach in detail.

The supervised autoencoder approach was initially proposed in Bottleneck encoder + MLP + Keras Tuner 8601c5, where one supervised autoencoder is trained separately before cross-validation (CV) split. I have realised that this training may cause label leakage because the autoencoder has seen part of the data in the validation set in each CV split and it can generate label-leakage features to overfit. So, my approach is to train the supervised autoencoder along with MLP in one model in each CV split. The training processes and explanations are given in the notebook and the following statements.

Cross-Validation (CV) Strategy and Feature Engineering:

5-fold 31-gap purged group time-series split
Remove first 85 days for training since they have different feature variance
Forward-fill the missing values
Transfer all resp targets (resp, resp_1, resp_2, resp_3, resp_4) to action for multi-label classification
Use the mean of the absolute values of all resp targets as sample weights for training so that the model can focus on capturing samples with large absolute resp.
During inference, the mean of all predicted actions is taken as the final probability
Deep Learning Model:

Use autoencoder to create new features, concatenating with the original features as the input to the downstream MLP model
Train autoencoder and MLP together in each CV split to prevent data leakage
Add target information to autoencoder (supervised learning) to force it to generate more relevant features, and to create a shortcut for backpropagation of gradient
Add Gaussian noise layer before encoder for data augmentation and to prevent overfitting
Use swish activation function instead of ReLU to prevent ‘dead neuron’ and smooth the gradient
Batch Normalisation and Dropout are used for MLP
Train the model with 3 different random seeds and take the average to reduce prediction variance
Only use the models (with different seeds) trained in the last two CV splits since they have seen more data
Only monitor the BCE loss of MLP instead of the overall loss for early stopping
Use Hyperopt to find the optimal hyperparameter set


flow chart:
        Input  features
         |
         |----------------------------------------
         v                                       |
   +-------------+                               |
   |GaussianNoise|                               |
   +-------------+                               |
         |                                       |
         v                                       |
   +-------------+                               |
   | Encoder     |                               |
   +-------------+                               |
         |                                       |
         |--------------------------------------||
         v                                      ||
   +------------+                               ||
   | Decoder    |                            +-------
   +------------+                            | MLP  |
         | ------------------|               +------+
         v                   |                  |
   +---------------------+   |                  v
   |Classification Layer |   |          Multi-Label Classification
   +---------------------+   |           (BCE Loss) Final Output
         |                   v       
         v              Regression(MSE Loss)         
Multi-Label Classification
(BCE Loss)
         |
         v
   +-----------+
   |   End     |
   +-----------+