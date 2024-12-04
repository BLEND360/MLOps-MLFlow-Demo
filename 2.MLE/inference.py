import mlflow
import mlflow.pyfunc
import boto3
import pandas as pd
import os
from io import StringIO

mlflow.set_tracking_uri("http://localhost:5000")  

from modules import EDA, utils, features, feature_importance, error_analysis, optimization, forecast

def inference_pipeline():

    # Step 2: Load the latest model from MLFlow
    model_name = "StoreCasting"
    model_version_alias = 'approved'
    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = mlflow.lightgbm.load_model(model_uri)
    

    train = pd.read_csv('input/train.csv', parse_dates=['date'])
    test = pd.read_csv('input/test.csv', parse_dates=['date'])
    df = pd.concat([train, test], sort=False) 

    df = features.feature_selection(train, df)
    print(df.columns)
    df = df.sort_values("date").reset_index(drop = True)

    cols = ['sales_lag_364', 'sales_lag_350', 'dayofweek_sales_lag_12', 'item_cluster', 'last_12months_sales_mean', 'sales_ewm_alpha_05_lag_365', 'day_of_week', 'dayofweek_sales_lag_13', 'season', 'sales_ewm_alpha_09_lag_91', 'store_cluster', 'sales_ewm_alpha_08_lag_91', 'last_12months_sales_sum', 'sales_ewm_alpha_07_lag_728', 'dayofweek_sales_lag_105', 'sales_ewm_alpha_05_lag_728', 'sales_ewm_alpha_095_lag_98', 'sales_ewm_alpha_07_lag_365', 'storesim_last_6months_sales_mean', 'month', 'sales_ewm_alpha_095_lag_91', 'last_3months_sales_mean', 'sales_ewm_alpha_07_lag_270', 'sales_ewm_alpha_05_lag_270', 'sales_ewm_alpha_07_lag_91', 'sales_ewm_alpha_095_lag_270', 'sales_lag_363', 'sales_ewm_alpha_08_lag_728', 'sales_ewm_alpha_095_lag_105', 'week_of_year', 'storesim_last_6months_sales_min', 'itemsim_last_9months_sales_std', 'dayofweek_sales_lag_40', 'is_wknd', 'day_of_year', 'sales_lag_700', 'itemsim_last_12months_sales_std', 'sales_lag_370', 'sales_ewm_alpha_08_lag_105', 'sales_ewm_alpha_08_lag_98', 'store_last_6months_sales_mean', 'itemsim_last_12months_sales_min', 'sales_ewm_alpha_09_lag_728', 'sales_ewm_alpha_07_lag_98', 'store_last_6months_sales_sum', 'sales_ewm_alpha_09_lag_98', 'dayofyear_sales_lag_2', 'sales_ewm_alpha_08_lag_112', 'sales_ewm_alpha_05_lag_105', 'last_3months_sales_sum', 'sales_lag_362', 'item_last_6months_sales_min', 'store_last_12months_sales_mean', 'sales_ewm_alpha_095_lag_112', 'itemsim_last_12months_sales_mean', 'ItemSalesSimilarity', 'sales_ewm_alpha_095_lag_728', 'item_last_6months_sales_std', 'sales_ewm_alpha_08_lag_365', 'dayofweek_sales_lag_112', 'itemsim_last_6months_sales_min', 'sales_ewm_alpha_07_lag_112', 'sales_ewm_alpha_07_lag_105', 'dayofyear_sales_lag_4', 'last_9months_sales_mean', 'storesim_last_12months_sales_sum']

    df.sort_values(["store", "item", "date"], inplace=True)
    train_final = df.loc[(df["date"] < "2018-01-01"), :]
    test_final = df.loc[(df["date"] >= "2018-01-01"), :]

    X_train_final = train_final[cols]
    Y_train_final = train_final.sales
    X_test_final = test_final[cols]

    for i in range(1,11):
        forecast.forecast_stores(train, X_test_final, test_final, model,store = i)
    
if __name__ == "__main__":
    inference_pipeline()
