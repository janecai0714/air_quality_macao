import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, kendalltau
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, Activation, LSTM, GRU
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from joblib import dump, load
import shap

time_step = 3
n_features = 35
name = 'SO2'  ###please input the column name of prediction
unit = '(ppm)' ###please input the unit
# unit = '(micro-gram/m3)' # for PM
place_and_time = 'Macau residential 2020-2021'
def data_prepare(train_dir, test_dir):
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
    train_data_X = train_data.iloc[:, :-1]
    train_data_Y = train_data.iloc[:, -1]

    test_data = pd.read_excel(test_dir)
    test_data['Datetime'] = pd.to_datetime(test_data[['YEAR', 'MONTH', 'DAY']])
    test_data.set_index('Datetime', inplace=True)
    test_data.drop(['YEAR', 'MONTH', 'DAY'], axis=1, inplace=True)
    test_data.dropna(inplace=True)
    test_data_X = test_data.iloc[:, :-1]
    test_data_Y = test_data.iloc[:, -1]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    train_X = scaler_X.fit_transform(train_data_X)
    #train_X = np.expand_dims(train_X, axis=2)
    train_Y = scaler_Y.fit_transform(train_data_Y.values.reshape(-1, 1))

    test_X = scaler_X.transform(test_data_X)
    #test_X = np.expand_dims(test_X, axis=2)
    test_Y = scaler_Y.transform(test_data_Y.values.reshape(-1, 1))

    data_for_plot = pd.DataFrame(index=test_data_Y.index)
    data_for_plot['measured_data' + '_' + name] = scaler_Y.inverse_transform(test_Y).ravel()
    return train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot

def rf_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot):
    train_X, train_Y, test_X, test_Y = train_X.squeeze(), train_Y.squeeze(), test_X.squeeze(), test_Y.squeeze()  # [num_sample, 35]
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=None)
    rf_model.fit(train_X, train_Y)
    pred_Y = rf_model.predict(test_X)
    real_pred_Y = scaler_Y.inverse_transform(pred_Y)
    real_test_Y = scaler_Y.inverse_transform(test_Y)
    rmse = mean_squared_error(real_test_Y, real_pred_Y, squared=False)
    mae = mean_absolute_error(real_test_Y, real_pred_Y)
    pcc = pearsonr(real_test_Y, real_pred_Y)[0]
    ktc = kendalltau(real_test_Y, real_pred_Y)[0]

    #rf_model.save('macao_residential_SO2_rf.h5')

    data_for_plot['predicted_data'+'_'+name] = real_pred_Y
    fig, ax = plt.subplots(figsize=(12, 8))  ## fig,ax = plt.subplots()
    data_for_plot.plot(y='measured_data' + '_' + name, use_index=True, ax=ax, fontsize=15)
    data_for_plot.plot(y='predicted_data' + '_' + name, use_index=True, ax=ax, linestyle='dashed', color='red',
                       fontsize=15)
    ax.set_title(f'{place_and_time} | Forecast vs Monitored | Random forest', fontsize=15)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel(f'{name}{unit}', fontsize=15)
    ax.tick_params(labelsize = 12)
    ax.tick_params(labelsize = 12)
    ax.legend(fontsize=12)
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_rf_timeline.svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(data_for_plot['measured_data' + '_' + name], data_for_plot['predicted_data' + '_' + name],
                s=10, color="lightgreen", edgecolors = "black")
    plt.xlabel(f'{name}_Measured', fontsize=15)
    plt.ylabel(f'{name}_Predicted', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'{place_and_time} | Scattered Graph | Random forest', fontsize=15)
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    x_points = [0, 12]
    y_points = [0, 12]
    plt.plot(x_points, y_points, linestyle='dashed', color='green')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_rf_scatter.svg")
    plt.close()

    # X_test_for_shap = pd.DataFrame(data=test_X, columns=train_data_X.columns)
    # explainer = shap.Explainer(rf_model.predict, X_test_for_shap)
    # #shap.initjs()
    # shap_values = explainer(X_test_for_shap)
    # shap.plots.bar(shap_values)
    # plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_rf_shap.svg")
    # plt.close()
    return rmse, mae, pcc, ktc

def svr_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot):
    train_X, train_Y, test_X, test_Y = train_X.squeeze(), train_Y.squeeze(), test_X.squeeze(), test_Y.squeeze()
    svr_model = SVR(kernel="linear")
    svr_model.fit(train_X, train_Y)
    pred_Y = svr_model.predict(test_X)
    real_pred_Y = scaler_Y.inverse_transform(pred_Y)
    real_test_Y = scaler_Y.inverse_transform(test_Y)
    rmse = mean_squared_error(real_test_Y, real_pred_Y, squared=False)
    mae = mean_absolute_error(real_test_Y, real_pred_Y)
    pcc = pearsonr(real_test_Y, real_pred_Y)[0]
    ktc = kendalltau(real_test_Y, real_pred_Y)[0]

    data_for_plot['predicted_data'+'_'+name] = real_pred_Y
    fig, ax = plt.subplots(figsize=(12, 8))  ## fig,ax = plt.subplots()
    data_for_plot.plot(y='measured_data' + '_' + name, use_index=True, ax=ax, fontsize=18)
    data_for_plot.plot(y='predicted_data' + '_' + name, use_index=True, ax=ax, linestyle='dashed', color='red',
                       fontsize=18)
    ax.set_title(f'{place_and_time} | Forecast vs Monitored | Support vector machine', fontsize=18)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel(f'{name}{unit}', fontsize=15)
    ax.tick_params(labelsize = 12)
    ax.tick_params(labelsize = 12)
    ax.legend(fontsize=12)
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_svr_timeline.svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(data_for_plot['measured_data' + '_' + name], data_for_plot['predicted_data' + '_' + name],
                s=10, color="lightgreen", edgecolors = "black")
    plt.xlabel(f'{name}_Measured', fontsize=15)
    plt.ylabel(f'{name}_Predicted', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'{place_and_time} | Scattered Graph | Support vector machine', fontsize=15)
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    x_points = [0, 12]
    y_points = [0, 12]
    plt.plot(x_points, y_points, linestyle='dashed', color='green')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_svr_scatter.svg")
    plt.close()
    return rmse, mae, pcc, ktc

def ann_train (train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot):
    print("ANN")
    train_X, test_X = train_X.squeeze(), test_X.squeeze() # [num_sample, 35]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(train_X.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    adam = tf.keras.optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mean_squared_error')
    epochs_hist = model.fit(train_X, train_Y, epochs=100, batch_size=32, validation_split=0.3)

    pred_Y = model.predict(test_X)
    real_pred_Y = scaler_Y.inverse_transform(pred_Y.squeeze())
    real_test_Y = scaler_Y.inverse_transform(test_Y.squeeze())
    rmse = mean_squared_error(real_test_Y, real_pred_Y, squared=False)
    mae = mean_absolute_error(real_test_Y, real_pred_Y)
    pcc = pearsonr(real_test_Y, real_pred_Y)[0]
    ktc = kendalltau(real_test_Y, real_pred_Y)[0]

    data_for_plot['predicted_data'+'_'+name] = real_pred_Y
    fig, ax = plt.subplots(figsize=(12, 8))  ## fig,ax = plt.subplots()
    data_for_plot.plot(y='measured_data' + '_' + name, use_index=True, ax=ax, fontsize=15)
    data_for_plot.plot(y='predicted_data' + '_' + name, use_index=True, ax=ax, linestyle='dashed', color='red',
                       fontsize=15)
    ax.set_title(f'{place_and_time} | Forecast vs Monitored | ANN', fontsize=15)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel(f'{name}{unit}', fontsize=15)
    ax.tick_params(labelsize = 12)
    ax.tick_params(labelsize = 12)
    ax.legend(fontsize=12)
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_ANN_timeline.svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(data_for_plot['measured_data' + '_' + name], data_for_plot['predicted_data' + '_' + name],
                s=10, color="lightgreen", edgecolors = "black")
    plt.xlabel(f'{name}_Measured', fontsize=15)
    plt.ylabel(f'{name}_Predicted', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'{place_and_time} | Scattered Graph | ANN', fontsize=15)
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    x_points = [0, 12]
    y_points = [0, 12]
    plt.plot(x_points, y_points, linestyle='dashed', color='green')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/SO2_ANN_scatter.svg")
    plt.close()
    return rmse, mae, pcc, ktc


def train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot, result_dir):
    model_name = ["RF", "SVR", "ANN"]
    pollutant_name = "SO2"
    num_repeat = 5
    repeat_list, model_list, pollutant_list, rmse_list, mae_list, pcc_list, ktc_list = [], [], [], [], [], [], []
    for i in range(num_repeat):
        for model in model_name:
            print("model:", model)
            if model == "RF":
                rmse, mae, pcc, ktc = rf_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot)
            if model == "SVR":
                rmse, mae, pcc, ktc = svr_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot)
            if model == "ANN":
                rmse, mae, pcc, ktc = ann_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot)
            repeat_list.append(i)
            model_list.append(model)
            pollutant_list.append(pollutant_name)
            rmse_list.append(rmse)
            mae_list.append(mae)
            pcc_list.append(pcc)
            ktc_list.append(ktc)
    result_dict = {"repeat":repeat_list, "pollutant":pollutant_list, "model":model_list,
                   "rmse":rmse_list, "mae":mae_list,
                   "pcc":pcc_list, "ktc": ktc_list}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(result_dir, index = False)
    return result_df

if __name__=="__main__":
    train_dir ="/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/SO2_2013_2019_tradition.xlsx"
    test_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/SO2_2020_2021_tradition.xlsx"
    train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot = data_prepare(train_dir, test_dir)
    # rmse, mae, pcc, ktc = rf_train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot)

    train_X_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/train_X_tradition.npy"
    train_Y_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/train_Y_tradition.npy"
    test_X_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/test_X_tradition.npy"
    test_Y_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/test_Y_tradition.npy"
    scaler_Y_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/scaler_Y_tradition.bin"
    train_data_X_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/train_data_tradition.pickle"
    data_for_plot_dir = "/home/jianxiu/Documents/air/deep/data/data_macao_residential/SO2/data_for_plot_tradition.pickle"
    result_dir = "/home/jianxiu/Documents/air/deep/result/result_macao_residential/SO2/macao_residential_SO2_tradition.csv"
    np.save(train_X_dir, train_X), np.save(train_Y_dir, train_Y), np.save(test_X_dir, test_X), np.save(test_Y_dir,
                                                                                                       test_Y)
    train_data_X.to_pickle(train_data_X_dir), data_for_plot.to_pickle(data_for_plot_dir)
    dump(scaler_Y, scaler_Y_dir)
    train_X, train_Y, test_X, test_Y = np.load(train_X_dir), np.load(train_Y_dir), np.load(test_X_dir), np.load(
        test_Y_dir)
    train_data_X, data_for_plot = pd.read_pickle(train_data_X_dir), pd.read_pickle(data_for_plot_dir)
    scaler_Y = load(scaler_Y_dir)


    result_df = train(train_X, train_Y, test_X, test_Y, scaler_Y, train_data_X, data_for_plot, result_dir)
    print("smart")


