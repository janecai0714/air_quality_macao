import pandas as pd
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, Activation, LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from tensorflow import keras
from main import *
from joblib import dump, load
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import shap

time_step = 3
n_features = 33
name = 'SO2'  ###please input the column name of prediction
unit = '(ppm)' ###please input the unit
# unit = '(micro-gram/m3)' # for PM
place_and_time = 'Macau residential 2020-2021'
def data_prepare(train_dir, test_dir, time_step):
    # load and clean train_data
    train_data = pd.read_excel(train_dir)
    train_data['Datetime'] = pd.to_datetime(train_data[['YEAR', 'MONTH', 'DAY']])
    train_data.set_index('Datetime', inplace=True)
    train_data.drop(['YEAR', 'MONTH', 'DAY'], axis=1, inplace=True)
    train_data.dropna(inplace=True)
    z_scores = stats.zscore(train_data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    train_data = train_data[filtered_entries]

    scaler_X = StandardScaler().fit(train_data)
    scaler_Y = StandardScaler()
    train_data_scale = scaler_X.transform(train_data)
    train_X = []
    train_Y = []
    for i in range(len(train_data_scale)-time_step):
        train_X.append(train_data_scale[i : i+time_step, :])
        train_Y.append(train_data.iloc[i+time_step, -1])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    train_Y = scaler_Y.fit_transform(train_Y.reshape(-1, 1))

    test_data = pd.read_excel(test_dir)
    test_data['Datetime'] = pd.to_datetime(test_data[['YEAR', 'MONTH', 'DAY']])
    test_data.set_index('Datetime', inplace=True)
    test_data.drop(['YEAR', 'MONTH', 'DAY'], axis=1, inplace=True)
    test_data.dropna(inplace=True)
    test_data_scale = scaler_X.transform(test_data)
    test_X = []
    test_Y = []
    for i in range(len(test_data_scale)-time_step):
        test_X.append(test_data_scale[i : i+time_step, :])
        test_Y.append(test_data.iloc[i+time_step, -1])
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    test_Y = scaler_Y.transform(test_Y.reshape(-1, 1))

    data_for_plot = pd.DataFrame(index = test_data.iloc[time_step-1:-1, -1].index)
    data_for_plot['measured_data'+'_'+name] = scaler_Y.inverse_transform(test_Y).ravel()
    return train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot

def lstm_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot):
    print("LSTM")
    regressor = Sequential()
    regressor.add(Dense(64, input_shape=(time_step, n_features)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=64, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=64, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=64))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    history = regressor.fit(train_X, train_Y, epochs=20, batch_size=32)

    pred_Y = regressor.predict(test_X)
    real_pred_Y = scaler_Y.inverse_transform(pred_Y.squeeze())
    real_test_Y = scaler_Y.inverse_transform(test_Y.squeeze())
    rmse = mean_squared_error(real_test_Y, real_pred_Y, squared=False)
    mae = mean_absolute_error(real_test_Y, real_pred_Y)
    pcc = pearsonr(real_test_Y, real_pred_Y)[0]
    ktc = kendalltau(real_test_Y, real_pred_Y)[0]
    regressor.save('macao_residential_SO2_LSTM.h5')

    data_for_plot['predicted_data'+'_'+name] = real_pred_Y
    fig, ax = plt.subplots(figsize=(12, 8))  ## fig,ax = plt.subplots()
    data_for_plot.plot(y='measured_data' + '_' + name, use_index=True, ax=ax, fontsize=15)
    data_for_plot.plot(y='predicted_data' + '_' + name, use_index=True, ax=ax, linestyle='dashed', color='red',
                       fontsize=15)
    ax.set_title(f'{place_and_time} | Forecast vs Monitored | LSTM', fontsize=15)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel(f'{name}{unit}', fontsize=15)
    ax.tick_params(labelsize = 12)
    ax.legend(fontsize=12)
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_LSTM_timeline.svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(data_for_plot['measured_data' + '_' + name], data_for_plot['predicted_data' + '_' + name],
                s=10, color="lightgreen", edgecolors = "black")
    plt.xlabel(f'{name}_Measured', fontsize=15)
    plt.ylabel(f'{name}_Predicted', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'{place_and_time} | Scattered Graph | LSTM', fontsize=15)
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    x_points = [0, 12]
    y_points = [0, 12]
    plt.plot(x_points, y_points, linestyle='dashed', color='green')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_LSTM_scatter.svg")
    plt.close()
    # X_test_for_shap = pd.DataFrame(data=test_X, columns=train_data.columns)
    # explainer = shap.Explainer(regressor.predict, X_test_for_shap)
    # shap.initjs()
    # shap_values = explainer(X_test_for_shap)
    # shap.plots.bar(shap_values)
    return rmse, mae, pcc, ktc

def rnn_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot):
    model = tf.keras.Sequential()
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(SimpleRNN(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(train_X, train_Y, epochs=100, batch_size=32)
    pred_Y = model.predict(test_X)
    real_pred_Y = scaler_Y.inverse_transform(pred_Y.squeeze())
    real_test_Y = scaler_Y.inverse_transform(test_Y.squeeze())
    rmse = mean_squared_error(real_test_Y, real_pred_Y, squared=False)
    mae = mean_absolute_error(real_test_Y, real_pred_Y)
    pcc = pearsonr(real_test_Y, real_pred_Y)[0]
    ktc = kendalltau(real_test_Y, real_pred_Y)[0]
    model.save('macao_residential_SO2_RNN.h5')

    data_for_plot['predicted_data' + '_' + name] = real_pred_Y
    fig, ax = plt.subplots(figsize=(12, 8))  ## fig,ax = plt.subplots()
    data_for_plot.plot(y='measured_data' + '_' + name, use_index=True, ax=ax, fontsize=15)
    data_for_plot.plot(y='predicted_data' + '_' + name, use_index=True, ax=ax, linestyle='dashed', color='red',
                       fontsize=15)
    ax.set_title(f'{place_and_time} | Forecast vs Monitored | RNN', fontsize=15)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel(f'{name}{unit}', fontsize=15)
    ax.tick_params(labelsize = 12)
    ax.tick_params(labelsize = 12)
    ax.legend(fontsize=12)
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_RNN_timeline.svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(data_for_plot['measured_data' + '_' + name], data_for_plot['predicted_data' + '_' + name],
                s=10, color="lightgreen", edgecolors="black")
    plt.xlabel(f'{name}_Measured', fontsize=15)
    plt.ylabel(f'{name}_Predicted', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'{place_and_time} | Scattered Graph | RNN', fontsize=15)
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    x_points = [0, 12]
    y_points = [0, 12]
    plt.plot(x_points, y_points, linestyle='dashed', color='green')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_RNN_scatter.svg")
    plt.close()
    return rmse, mae, pcc, ktc

def gru_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot):
    model_name = 'GRU'
    regressor = Sequential()
    regressor.add(GRU(units=64, return_sequences=True, input_shape=(3, 33)))
    regressor.add(Dropout(0.2))
    regressor.add(GRU(units=64, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(GRU(units=64, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(GRU(units=64))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    history = regressor.fit(train_X, train_Y, epochs=100, batch_size=32)
    pred_Y = regressor.predict(test_X)
    real_pred_Y = scaler_Y.inverse_transform(pred_Y.squeeze())
    real_test_Y = scaler_Y.inverse_transform(test_Y.squeeze())
    rmse = mean_squared_error(real_test_Y, real_pred_Y, squared=False)
    mae = mean_absolute_error(real_test_Y, real_pred_Y)
    pcc = pearsonr(real_test_Y, real_pred_Y)[0]
    ktc = kendalltau(real_test_Y, real_pred_Y)[0]
    regressor.save('macao_residential_SO2_GRU.h5')

    data_for_plot['predicted_data' + '_' + name] = real_pred_Y
    fig, ax = plt.subplots(figsize=(12, 8))  ## fig,ax = plt.subplots()
    data_for_plot.plot(y='measured_data' + '_' + name, use_index=True, ax=ax, fontsize=15)
    data_for_plot.plot(y='predicted_data' + '_' + name, use_index=True, ax=ax, linestyle='dashed', color='red',
                       fontsize=15)
    ax.set_title(f'{place_and_time} | Forecast vs Monitored | GRU', fontsize=15)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel(f'{name}{unit}', fontsize=15)
    ax.tick_params(labelsize = 12)
    ax.legend(fontsize=12)
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_GRU_timeline.svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(data_for_plot['measured_data' + '_' + name], data_for_plot['predicted_data' + '_' + name],
                s=10, color="lightgreen", edgecolors="black")
    plt.xlabel(f'{name}_Measured', fontsize=15)
    plt.ylabel(f'{name}_Predicted', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'{place_and_time} | Scattered Graph | GRU', fontsize=15)
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    x_points = [0, 12]
    y_points = [0, 12]
    plt.plot(x_points, y_points, linestyle='dashed', color='green')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_GRU_scatter.svg")
    plt.close()
    return rmse, mae, pcc, ktc

def train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot, result_dir):
    num_repeat = 5
    model_name = ["RNN", "LSTM", "GRU"]
    pollutant_name = "SO2"
    repeat_list, model_list, pollutant_list, rmse_list, mae_list, pcc_list, ktc_list = [], [], [], [], [], [], []
    for i in range(num_repeat):
        for model in model_name:
            print("model:", model)
            if model == "RNN":
                rmse, mae, pcc, ktc = rnn_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot)
            elif model == "LSTM":
                rmse, mae, pcc, ktc = lstm_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot)
            elif model == "GRU":
                rmse, mae, pcc, ktc = gru_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot)
            repeat_list.append(i)
            model_list.append(model)
            pollutant_list.append(pollutant_name)
            rmse_list.append(rmse)
            mae_list.append(mae)
            pcc_list.append(pcc)
            ktc_list.append(ktc)
    result_dict = {"repeat": repeat_list, "pollutant":pollutant_list, "model":model_list,
                   "rmse":rmse_list, "mae":mae_list,
                   "pcc":pcc_list, "ktc": ktc_list}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(result_dir, index=False)
    return result_df

if __name__=="__main__":
    train_dir ="/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/SO2_2013_2019.xlsx"
    test_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/SO2_2020_2021.xlsx"
    time_step = 3
    train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot = data_prepare(train_dir, test_dir, time_step)

    train_X_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/train_X.npy"
    train_Y_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/train_Y.npy"
    test_X_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/test_X.npy"
    test_Y_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/test_Y.npy"
    scaler_Y_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/scaler.npy"
    train_data_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/train_data.pickle"
    data_for_plot_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/data_for_plot.pickle"
    result_dir = "/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/macao_residential_SO2_dl.csv"
    np.save(train_X_dir, train_X), np.save(train_Y_dir, train_Y), np.save(test_X_dir, test_X), np.save(test_Y_dir, test_Y)
    train_data.to_pickle(train_data_dir), data_for_plot.to_pickle(data_for_plot_dir)
    dump(scaler_Y, scaler_Y_dir)

    train_X, train_Y, test_X, test_Y = \
        (np.load(train_X_dir), np.load(train_Y_dir), np.load(test_X_dir), np.load(test_Y_dir))
    scaler_Y = load(scaler_Y_dir)
    train_data, data_for_plot = pd.read_pickle(train_data_dir), pd.read_pickle(data_for_plot_dir)
    # rmse, mae, pcc, ktc = rnn_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot)
    result_df = train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data, data_for_plot, result_dir)
print('smart')