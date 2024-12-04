# Base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.lightgbm
import os

# Model
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

# Configuration
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# Module to include all EDA operations

def number_of_records(train, test):
    print(f"There are {train.store.nunique()} unique stores in train data")
    print(f"There are {test.store.nunique()} unique stores in test data")
    print(f"There are {train.item.nunique()} unique items in train data")
    print(f"There are {test.item.nunique()} unique items in test data")

    # Minimum and maximum dates for train and test data
    train_start, train_end = train['date'].min(), train['date'].max()
    test_start, test_end = test['date'].min(), test['date'].max()

    print(f"The date range in Training Data is {train_start} - {train_end}")
    print(f"The date range in Test Data is {test_start} - {test_end}")

    # Log parameters in MLFlow
    mlflow.log_param("train_start_date", train_start)
    mlflow.log_param("train_end_date", train_end)
    mlflow.log_param("test_start_date", test_start)
    mlflow.log_param("test_end_date", test_end)

def eda_metrics(df):
    # Calculate metrics
    items_in_store = df.groupby(["store"])["item"].nunique().reset_index(name="unique_items")
    store_sales_aggregate = df.groupby(["store"]).agg({
        "sales": ["count", "sum", "mean", "median", "std", "min", "max"]
    }).reset_index()
    
    item_sales_aggregate = df.groupby(["item"]).agg({
        "sales": ["count", "sum", "mean", "median", "std", "min", "max"]
    }).reset_index()

    store_sales_aggregate.columns = ['_'.join(col).strip() for col in store_sales_aggregate.columns.values]
    item_sales_aggregate.columns = ['_'.join(col).strip() for col in item_sales_aggregate.columns.values]

    # Create a Pandas Excel writer using XlsxWriter as the engine
    output_file = "artifacts/eda/eda_metrics.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet
        items_in_store.to_excel(writer, sheet_name='Items_in_Store', index=False)
        store_sales_aggregate.to_excel(writer, sheet_name='Store_Sales_Aggregate', index=False)
        item_sales_aggregate.to_excel(writer, sheet_name='Item_Sales_Aggregate', index=False)

    print("DataFrames have been written to 'eda_metrics.xlsx'")

    # Log the Excel file as an artifact in MLFlow
    mlflow.log_artifact(output_file)

def eda_store_sale_histogram(train):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(1, 11):
        if i < 6:
            train[train.store == i].sales.hist(ax=axes[0, i - 1])
            axes[0, i - 1].set_title("Store " + str(i), fontsize=15)
        else:
            train[train.store == i].sales.hist(ax=axes[1, i - 6])
            axes[1, i - 6].set_title("Store " + str(i), fontsize=15)
    
    plt.tight_layout(pad=4.5)
    plt.suptitle("Histogram: Sales")

    # Save the histogram as a PNG file
    output_file = "artifacts/eda/store_sales_histogram.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format="png")
    plt.close()

    # Log the histogram as an artifact in MLFlow
    mlflow.log_artifact(output_file)

def eda_store_item_histogram(train, store=1):
    sub = train[train.store == store].set_index("date")

    fig, axes = plt.subplots(10, 5, figsize=(20, 35))
    for i in range(1, 51):
        sub[sub.item == i].sales.plot(ax=axes[(i - 1) // 5, (i - 1) % 5], legend=True, label=f"Item {i} Sales")
    
    plt.tight_layout(pad=4.5)
    plt.suptitle(f"Store {store} Item Sales")

    output_file = "artifacts/eda/store_item_histogram.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format="png")
    plt.close()

    # Log the histogram as an artifact in MLFlow
    mlflow.log_artifact(output_file)

def eda_correlation_sales(train):
    storesales = train.groupby(["date", "store"]).sales.sum().reset_index().set_index("date")
    corr = pd.pivot_table(storesales, values="sales", columns="store", index="date").corr(method="spearman")
    
    plt.figure(figsize=(7, 7))
    sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 
                cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                annot=True, annot_kws={"size": 9}, square=True)
    
    output_file = "artifacts/eda/sales_correlation_heatmap.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format="png")
    plt.close()

    # Log the heatmap as an artifact in MLFlow
    mlflow.log_artifact(output_file)

def eda(train, test, df):
    mlflow.start_run(run_name = "EDA",nested = True)  # Start an MLFlow run
    number_of_records(train, test)
    eda_metrics(df)
    eda_store_sale_histogram(train)
    eda_store_item_histogram(train)
    eda_correlation_sales(train)

    mlflow.end_run()  # End the MLFlow run
