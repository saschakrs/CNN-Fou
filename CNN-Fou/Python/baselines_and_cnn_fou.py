# Required libraries
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp
import scipy.fftpack
from keras.layers import Dropout, Input, Dense, Flatten, Activation, LSTM, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from math import sqrt, floor
from pyramid.arima import auto_arima
from scipy.signal import periodogram, welch
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Preprocessing for ARIMA/ETS
def standardize_and_split_arima_ets(dat, train_ratio = 0.9, seqlen=50):
    train_lim = floor(len(dat)*(1-train_ratio))-seqlen-1 # sequence index of last train sequence
    train_lim_element = floor(len(dat)*(1-train_ratio)) # sequence index of last train sequence's last element
    train = dat[:train_lim]
    test = dat[train_lim:]
    sd = np.std(dat[:train_lim_element])
    m = np.mean(dat[:train_lim_element])
    train = (train-m)/sd
    test = (test-m)/sd
    return train, test
​
def fc_arima(dat, fc_horizon):
    arima = auto_arima(dat, start_p=1, start_q=1, max_d=2,
                 max_p=3, max_q=3, m=12,
                 start_P=0, seasonal=True,
                 D=1, trace=True,
                 error_action='ignore',  
                 suppress_warnings=True, 
                 stepwise=True)
    preds = arima.predict(n_periods=fc_horizon)
    return preds
​
# Evaluate ARIMA Forecasts (RMSE)
def evaluate_arima(train, test):
    preds = fc_arima(train, len(test))
    rmse = sqrt(mean_squared_error(test, preds))
    print("ARIMA RMSE:", np.round(rmse, 3))
    return preds, rmse
              
def fc_ets(dat, fc_horizon):
    model = ExponentialSmoothing(dat)
    model_fit = model.fit()
    preds = model_fit.forecast(fc_horizon)
    return preds
​
# Evaluate ETS Forecasts (RMSE)
def evaluate_ets(train, test):
    preds = fc_ets(train, len(test))
    rmse = sqrt(mean_squared_error(test, preds))
    print("ETS RMSE:", np.round(rmse, 3))
    return preds, rmse
	
	
# Sequence generation for neural networks
def generate_sequences(d, seqlen=50, rolling=False):
    X = []
    y = []
    for i in range(len(d)-2*seqlen-1):
        X.append(d[i:(i+seqlen)])
        y.append(d[(i+seqlen):(i+2*seqlen)])
    X, y = np.array(X), np.array(y)
    if rolling:
        y = y[:, 0]
        return X.reshape((X.shape[0], X.shape[1], 1)), y
    return X.reshape((X.shape[0], X.shape[1], 1)), y.reshape((y.shape[0], y.shape[1], 1))
​
# Generate train and test data for CNN/LSTM
def standardize_and_split_cnn_lstm(d, num_fc=10, train_ratio=0.9, seqlen=50, reverse=False, rolling=False):
    if reverse:
        d = list(reversed(d))
    train_lim = floor(train_ratio*len(d))
    d_mean = np.mean(d[:(train_lim-1)])
    d_sd = np.std(d[:(train_lim-1)])
    d = (d-d_mean)/d_sd
    
    X, y = generate_sequences(d, seqlen, rolling=rolling)
    X_train, y_train = generate_sequences(d[:(train_lim-num_fc)], seqlen, rolling=rolling)
    X_test, y_test = generate_sequences(d[(train_lim-num_fc):], seqlen, rolling=rolling)
    return X_train, y_train, X_test, y_test, d_mean, d_sd
​
# Determine strongest Fourier periodicities
def get_periodicity_fourier(series, k=2, seqlen_min=1, seqlen_max=1000):
    perio = periodogram(series, fs=1)
    perio_df = pd.DataFrame({"freq": perio[0], "spec": perio[1]})
    perio_df.sort_values(by="spec", ascending=False, inplace=True) # sort decreasingly
    top_freqs = perio_df["freq"].values
    top_lags = np.array([round(1/x) for x in top_freqs if x!=0])
    # unique lags only
    _, idx = np.unique(top_lags, return_index=True)
    top_k_lags = top_lags[np.sort(idx)]
    top_k_lags = np.int_(top_k_lags)
    top_k_lags = top_k_lags# [:k]
    top_k_lags = top_k_lags[top_k_lags>=seqlen_min]
    top_k_lags = top_k_lags[top_k_lags<=seqlen_max]
    top_k_lags = top_k_lags[:k]
    if len(top_k_lags) < k:
        print("Warning: Too little Fourier seqlens available.")
    return top_k_lags
​
# Train and evaluate standard LSTM (RMSE)
def evaluate_lstm(d, seqlen=50, num_fc=10, epochs=100, batch_size=16, rolling=False):
    model = Sequential()
    model.add(LSTM(input_shape=(None, 1), units=64, return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
​
    # Only forecast one point to the future in case of rolling forecast
    X_train, y_train, X_test, y_test, d_mean, d_sd = standardize_and_split_cnn_lstm(d, seqlen=seqlen, rolling=rolling, train_ratio=.9)
    # Reshaping required if rolling is True
    if rolling:
        y_train = y_train.reshape((y_train.shape[0], 1, 1))
        y_test = y_test.reshape((y_test.shape[0], 1, 1))
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    
    # Evaluate on test data
    yhat_test = model.predict(X_test)
    # Only consider first forecast in case of rolling forecast. Otherwise, evaluate on sequences of length num_fc.
    if rolling:
        yhat_test = yhat_test[:, 0]
    else:
        yhat_test = yhat_test[:, :num_fc]
        y_test = y_test[:, :num_fc]
​
    # Reshape for mse function
    yhat_test = yhat_test.reshape(yhat_test.shape[:2])
    y_test = y_test.reshape(y_test.shape[:2])
​
    curr_rmse = sqrt(mean_squared_error(y_test, yhat_test))
    print("RMSE:", curr_rmse)
    return y_test, yhat_test, curr_rmse
 
# Train and evaluate CNN with integrated Fourier parametrization (RMSE)
def evaluate_cnn(d, seqlen=50, num_fc=10, epochs=100, batch_size=16, nb_filter=96, rolling=False, verbose=1, std_filters=False):
    X_train, y_train, X_test, y_test, d_mean, d_sd = standardize_and_split_cnn_lstm(d, seqlen=seqlen, rolling=True, train_ratio=.9, num_fc=num_fc)
    window_size=seqlen
    
    if std_filters is False:
        # Get Fourier parameters
        filter_lengths = get_periodicity_fourier(d)
    else:
        filter_lengths = [2, 8]
      
      
    
    nb_samples, nb_series = (X_train.shape[0], 1)
​
    model = Sequential()
    model.add(Conv1D(filters=nb_filter, input_shape = (window_size, nb_series),
                              kernel_size=filter_lengths[0],
                              activation='relu',
                              strides=1,
                              padding="causal"))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D(filters=nb_filter,
                     kernel_size=filter_lengths[1],
                     padding='causal',
                     activation='relu',
                     strides=1))
    
    model.add(MaxPooling1D())
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss="mse", optimizer=Adam(lr=0.0001))
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=verbose)
    # Evaluate on test data   
    yhat_test = model.predict(X_test)
    # CNN performs rolling forecasts (only 1 point ahead) by definition
    # Use the indirect method for multi step: Prediction serves as next input (rolling=False)
    if not rolling:
        # update test data such that it applies to multi step
        y_test = standardize_and_split(d, seqlen=seqlen, rolling=False, train_ratio=.9)[3]
        y_test = y_test[:, :num_fc]
        y_test = y_test.reshape(y_test.shape[:2])
        for fc_step in np.arange(1, num_fc):
            # Update X_test one step forward (i.e., remove first value of X_test and append yhat_test as last element)
            X_test = X_test[:, 1:] # remove all the first elements
            X_test = np.append(X_test, yhat_test[:, -1].reshape((yhat_test.shape[0], 1, 1)), axis=1)
            current_yhat_test = model.predict(X_test)
            # Extend forecasts
            yhat_test = np.append(yhat_test, current_yhat_test, axis=1)
    curr_rmse = sqrt(mean_squared_error(y_test, yhat_test))
    print("RMSE:", curr_rmse)
    return y_test, yhat_test, curr_rmse
	
	
# Data preprocessing for LSTM Snapshot ensembles
# Unlike the previous methods, Snapshot ensembles require a meta training set to learn the ensemble weights of the base learners
def standardize_and_split_snapshot_ensemble(d, train_ratio = .9, meta_train_ratio = .9, seqlen = 50, reverse = False):
    if reverse:
        d = list(reversed(d))
    train_lim = int(train_ratio*len(d))
    meta_train_lim = int(train_lim+(len(d)-train_lim)*meta_train_ratio)
    d_mean = d[:train_lim].mean()
    d_sd = d[:train_lim].std()
    d = (d-d_mean)/d_sd
    
    X_train, y_train = generate_sequences(d[:train_lim], seqlen)
    X_meta_train, y_meta_train = generate_sequences(d[train_lim:meta_train_lim], seqlen)
    X_test, y_test = generate_sequences(d[meta_train_lim:], seqlen)
    return X_train, y_train, X_meta_train, y_meta_train, X_test, y_test, d_mean, d_sd
​
# Train and evaluate Snapshot LSTM (RMSE)
def evaluate_snapshot_lstm(d, d_name, seqlens = np.arange(10, 101, 10), num_fc = 10, epochs_per_seqlen=10, batch_size=16,
                       load_pretrained = None):
    if num_fc is None:
        num_fc = min(seqlens)
    # By default, a new snapshot is trained based on random init weights
    if load_pretrained is None:
        model = Sequential()
        model.add(LSTM(input_shape=(None, 1), units=64, return_sequences=True))
        model.add(Dropout(.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(.2))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
    # To specify a pretrained model, set load_pretrained accordingly    
    else:
        # Load most recent model state and continue training
        # model = load_model('..')
    rmses = {}
    for seqlen in seqlens:
        print("seqlen: "+str(seqlen)+"=====")
        X_train, y_train, X_meta_train, y_meta_train, X_test, y_test, d_mean, d_sd = standardize_and_split_snapshot_ensemble(d, seqlen=seqlen)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_per_seqlen, validation_split=0.1, 
                  verbose=False)
        model.save('models/'+d_name+'/snapshots/snapshot_'+str(seqlen)+'.h5')
        # Evaluate on test data
        yhat_meta_train = model.predict(X_meta_train)[:, :num_fc]
        curr_rmse = np.sqrt((((yhat_meta_train*d_sd+d_mean) - (y_meta_train[:, :num_fc]*d_sd+d_mean)) ** 2).mean())
        print("RMSE of Snap", seqlen, ":", curr_rmse)
        rmses["snap"+str(seqlen)] = curr_rmse
    return rmses
