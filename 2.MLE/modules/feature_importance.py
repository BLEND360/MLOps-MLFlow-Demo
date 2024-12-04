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


def plot_lgb_importances(model, plot=False, num=10):
    from matplotlib import pyplot as plt
    import seaborn as sns
    
    gain = model.booster_.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name_,
                             'split': model.booster_.feature_importance(importance_type='split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    
    # Save the feature importance to an Excel file and log as an artifact
    with pd.ExcelWriter("artifacts/features/feature_importance.xlsx", engine='xlsxwriter') as writer:
        feat_imp.to_excel(writer, sheet_name='Feature Importance', index=False)

    mlflow.log_artifact("artifacts/features/feature_importance.xlsx")

    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('Top 30 Features by Gain')
        plt.tight_layout()
        plt.savefig("artifacts/features/top_30_features.png", format="png")
        plt.close()
        
        # Log the feature importance plot as an artifact
        mlflow.log_artifact("artifacts/features/top_30_features.png")
    else:
        return feat_imp


def shap_plots(model, X_train, X_val):
    explainer = shap.Explainer(model)
    shap_values_train = explainer(X_train)
    shap_values_valid = explainer(X_val)

    # SHAP Beeswarm plot for training set
    plt.figure()
    shap_values_train = shap.plots.beeswarm(shap_values_train, max_display=30)
    plt.savefig("artifacts/features/shap_beeswarm_plot_train.png", format="png")
    plt.close(shap_values_train)
    mlflow.log_artifact("artifacts/features/shap_beeswarm_plot_train.png")

    # SHAP Beeswarm plot for validation set
    plt.figure()
    shap_values_valid = shap.plots.beeswarm(shap_values_valid, max_display=30)
    plt.savefig("artifacts/features/shap_beeswarm_plot_valid.png", format="png")
    plt.close(shap_values_valid)
    mlflow.log_artifact("artifacts/features/shap_beeswarm_plot_valid.png")

    # SHAP Bar plot for training set
    plt.figure()
    shap_values_train = shap.plots.bar(shap_values_train, max_display=30)
    plt.savefig("artifacts/features/shap_bar_plot_train.png", format="png")
    plt.close(shap_values_train)
    mlflow.log_artifact("artifacts/features/shap_bar_plot_train.png")


def feature_importance(first_model, X_train, X_val):
    
    # Log feature importance and SHAP plots
    feature_imp_df = plot_lgb_importances(first_model, num=50)
    plot_lgb_importances(first_model, plot=True, num=30)
    #shap_plots(first_model, X_train, X_val)
    return feature_imp_df  

