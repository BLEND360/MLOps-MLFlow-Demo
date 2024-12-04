import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

# Model
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

# Configuration
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def validation_error_analysis(val, X_val, Y_val, first_model):
    error = pd.DataFrame({
        "date": val.date,
        "store": X_val.store,
        "item": X_val.item,
        "actual": Y_val,
        "pred": first_model.predict(X_val)
    }).reset_index(drop=True)

    error["error"] = np.abs(error.actual - error.pred)
    error.sort_values("error", ascending=False).head(20)

    error_stats = error[["actual", "pred", "error"]].describe([0.7, 0.8, 0.9, 0.95, 0.99]).T

    mean_store_item_error = error.groupby(["store", "item"]).error.mean().sort_values(ascending=False)
    mean_store_error = error.groupby(["store"]).error.mean().sort_values(ascending=False)
    mean_item_error = error.groupby(["item"]).error.mean().sort_values(ascending=False)

    error_file = "artifacts/error_analysis/error_metrics.xlsx"
    with pd.ExcelWriter(error_file, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet
        error.to_excel(writer, sheet_name='Val Error', index=False)
        error_stats.to_excel(writer, sheet_name='Error Stats', index=False)
        mean_store_item_error.to_excel(writer, sheet_name='Mean Store Item Error', index=False)
        mean_store_error.to_excel(writer, sheet_name='Mean store error', index=False)
        mean_item_error.to_excel(writer, sheet_name='Mean item error', index=False)

    # Log error metrics Excel file as an artifact
    mlflow.log_artifact(error_file)
    
    return error


def actual_prediction_plot_store(error, store=1):
    # Store 1 Actual - Pred
    sub = error[error.store == store].set_index("date")
    fig, axes = plt.subplots(10, 5, figsize=(20, 35))
    for i in range(1, 51):
        if i < 6:
            sub[sub.item == i].actual.plot(ax=axes[0, i - 1], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[0, i - 1], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 6 and i < 11:
            sub[sub.item == i].actual.plot(ax=axes[1, i - 6], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[1, i - 6], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 11 and i < 16:
            sub[sub.item == i].actual.plot(ax=axes[2, i - 11], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[2, i - 11], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 16 and i < 21:
            sub[sub.item == i].actual.plot(ax=axes[3, i - 16], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[3, i - 16], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 21 and i < 26:
            sub[sub.item == i].actual.plot(ax=axes[4, i - 21], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[4, i - 21], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 26 and i < 31:
            sub[sub.item == i].actual.plot(ax=axes[5, i - 26], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[5, i - 26], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 31 and i < 36:
            sub[sub.item == i].actual.plot(ax=axes[6, i - 31], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[6, i - 31], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 36 and i < 41:
            sub[sub.item == i].actual.plot(ax=axes[7, i - 36], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[7, i - 36], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 41 and i < 46:
            sub[sub.item == i].actual.plot(ax=axes[8, i - 41], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[8, i - 41], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")
        if i >= 46 and i < 51:
            sub[sub.item == i].actual.plot(ax=axes[9, i - 46], legend=True, label="Item " + str(i) + " Sales")
            sub[sub.item == i].pred.plot(ax=axes[9, i - 46], legend=True, label="Item " + str(i) + " Pred", linestyle="dashed")

    plt.tight_layout(pad=4.5)
    plt.suptitle("Store 1 Item Sales Diagram")
    plot_file = f"artifacts/error_analysis/actual_vs_predicted_store{store}.png"
    plt.savefig(plot_file, format="png")
    plt.close()

    # Log the plot as an artifact
    mlflow.log_artifact(plot_file)


def error_analysis_plots(error):
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    for axi in axes.flat:
        axi.ticklabel_format(style="sci", axis="y", scilimits=(0, 10))
        axi.ticklabel_format(style="sci", axis="x", scilimits=(0, 10))
        axi.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        axi.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    (error.actual - error.pred).hist(ax=axes[0, 0], color="steelblue", bins=20)
    error.error.hist(ax=axes[0, 1], color="steelblue", bins=20)
    sr = error.copy()
    sr["StandardizedR"] = (sr.error / (sr.actual - sr.pred).std())
    sr["StandardizedR2"] = ((sr.error / (sr.actual - sr.pred).std())**2)
    sr.plot.scatter(x="pred", y="StandardizedR", color="red", ax=axes[1, 0])
    sr.plot.scatter(x="pred", y="StandardizedR2", color="red", ax=axes[1, 1])
    error.actual.hist(ax=axes[2, 0], color="purple", bins=20)
    error.pred.hist(ax=axes[2, 1], color="purple", bins=20)
    error.plot.scatter(x="actual", y="pred", color="seagreen", ax=axes[3, 0])

    # QQ Plot
    import statsmodels.api as sm
    sm.qqplot(sr.pred, ax=axes[3, 1], c="seagreen")

    plt.suptitle("ERROR ANALYSIS", fontsize=20)
    axes[0, 0].set_title("Error Histogram", fontsize=15)
    axes[0, 1].set_title("Absolute Error Histogram", fontsize=15)
    axes[1, 0].set_title("Standardized Residuals & Fitted Values", fontsize=15)
    axes[1, 1].set_title("Standardized Residuals^2 & Fitted Values", fontsize=15)
    axes[2, 0].set_title("Actual Histogram", fontsize=15)
    axes[2, 1].set_title("Predicted Histogram", fontsize=15)
    axes[3, 0].set_title("Actual vs Predicted Scatter Plot", fontsize=15)
    axes[3, 1].set_title("Normal Q-Q Plot of Standardized Residuals", fontsize=15)

    plot_file = "artifacts/error_analysis/error_analysis_plots.png"
    plt.savefig(plot_file, format="png")
    plt.close()

    # Log the plot as an artifact
    mlflow.log_artifact(plot_file)

# Track experiment in MLflow

def error_analysis(val,X_val,Y_val,first_model):
    error = validation_error_analysis(val,X_val,Y_val,first_model)
    actual_prediction_plot_store(error)
    error_analysis_plots(error)