import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# Model
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

# Configuration
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def forecast_stores(train, X_test_final, test_final, final_model, store=1):
    # Forecast prediction    
    forecast = pd.DataFrame({
        "date": test_final.date,
        "store": test_final.store,
        "item": test_final.item,
        "sales": final_model.predict(X_test_final)
    })

    # Subset for the specified store
    sub = train[train.store == store].set_index("date")
    forc = forecast[forecast.store == store].set_index("date")

    # Plot forecast vs actual for items in the store
    fig, axes = plt.subplots(10, 5, figsize=(20, 35))
    for i in range(1, 51):
        if i < 6:
            sub[sub.item == i].sales.plot(ax=axes[0, i - 1], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[0, i - 1], legend=True, label="Forecast")
        if 6 <= i < 11:
            sub[sub.item == i].sales.plot(ax=axes[1, i - 6], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[1, i - 6], legend=True, label="Forecast")
        if 11 <= i < 16:
            sub[sub.item == i].sales.plot(ax=axes[2, i - 11], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[2, i - 11], legend=True, label="Forecast")
        if 16 <= i < 21:
            sub[sub.item == i].sales.plot(ax=axes[3, i - 16], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[3, i - 16], legend=True, label="Forecast")
        if 21 <= i < 26:
            sub[sub.item == i].sales.plot(ax=axes[4, i - 21], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[4, i - 21], legend=True, label="Forecast")
        if 26 <= i < 31:
            sub[sub.item == i].sales.plot(ax=axes[5, i - 26], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[5, i - 26], legend=True, label="Forecast")
        if 31 <= i < 36:
            sub[sub.item == i].sales.plot(ax=axes[6, i - 31], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[6, i - 31], legend=True, label="Forecast")
        if 36 <= i < 41:
            sub[sub.item == i].sales.plot(ax=axes[7, i - 36], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[7, i - 36], legend=True, label="Forecast")
        if 41 <= i < 46:
            sub[sub.item == i].sales.plot(ax=axes[8, i - 41], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[8, i - 41], legend=True, label="Forecast")
        if 46 <= i < 51:
            sub[sub.item == i].sales.plot(ax=axes[9, i - 46], legend=True, label="Item " + str(i) + " Sales")
            forc[forc.item == i].sales.plot(ax=axes[9, i - 46], legend=True, label="Forecast")

    # Layout adjustments
    plt.tight_layout(pad=6.5)
    plt.suptitle(f"Store {store} Items Actual & Forecast")

    # Save and log the plot
    plot_path = f"artifacts/forecast/store_item_forecast_store{store}.png"
    plt.savefig(plot_path, format="png")
    plt.close()

    # Log the plot as an MLflow artifact
    mlflow.log_artifact(plot_path)

