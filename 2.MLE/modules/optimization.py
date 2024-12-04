import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

# Configuration
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def hyperparameter_tuning(smape,X_train,Y_train,cols):
    lgbm_params = {
        
        "num_leaves":[20,31], # Default 31
        "max_depth":[-1, 20, 30], # Default -1
        "learning_rate":[0.1, 0.05], # Default 0.1
        "n_estimators":[10000,15000], # Default 100
        "min_split_gain":[0.0, 2,5], # Default 0
        "min_child_samples":[10, 20, 30], # Default 20
        "colsample_bytree":[0.5, 0.8, 1.0], # Default 1
        "reg_alpha":[0.0, 0.5, 1], # Default 0
        "reg_lambda":[0.0, 0.5, 1] # Default 0
    }

    model = lgb.LGBMRegressor(random_state=384)



    tscv = TimeSeriesSplit(n_splits=3)

    rsearch = RandomizedSearchCV(model, lgbm_params, random_state=384, 
                                cv=tscv, scoring=make_scorer(smape),
                                verbose = True, n_jobs = -1).fit(
        X_train[cols], Y_train
    )

    return rsearch.best_params_


def num_iteration(X_train,Y_train,cols,lgbm_smape,X_val,Y_val,best_params):
    model_tuned2 = lgb.LGBMRegressor(**best_params, random_state=384, metric = "custom")
              
    model_tuned2.fit(
        X_train[cols], Y_train,
        eval_metric= lambda y_true, y_pred: [lgbm_smape(y_true, y_pred)],
        eval_set = [(X_train[cols], Y_train), (X_val[cols], Y_val)],
        eval_names = ["Train", "Valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=1000),lgb.log_evaluation(period=500)  ]
    )
    print("Best Iteration:", model_tuned2.booster_.best_iteration)
    return model_tuned2.booster_.best_iteration


