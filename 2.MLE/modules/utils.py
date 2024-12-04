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


def train_validation_split(df):
    # Dataframe must be sorted by date because of Time Series Split 
    df = df.sort_values("date").reset_index(drop = True)

    # Train Validation Split
    # Validation set includes 3 months (Oct. Nov. Dec. 2017)
    train = df.loc[(df["date"] < "2017-10-01"), :]
    val = df.loc[(df["date"] >= "2017-10-01") & (df["date"] < "2018-01-01"), :]


    cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

    Y_train = train['sales']
    X_train = train[cols]

    Y_val = val['sales']
    X_val = val[cols]

    return X_train,X_val,Y_train,Y_val,train,val,df

# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def lgbm_smape(y_true, y_pred):
    smape_val = smape(y_true, y_pred)
    return 'SMAPE', smape_val, False