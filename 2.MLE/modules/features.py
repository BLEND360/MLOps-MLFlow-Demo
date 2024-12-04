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

def CompareTwoGroups(dataframe, group, target):
    
    import itertools
    from scipy.stats import shapiro
    import scipy.stats as stats
    
    # 1. Normality Test: Shapiro Test
    # 2. Homogeneity Test: Levene Test
    # 3. Parametric or Non-Parametric T Test: T-Test, Welch Test, Mann Whitney U
    
    # Create Combinations
    item_comb = list(itertools.combinations(dataframe[group].unique(), 2))
    
    AB = pd.DataFrame()
    for i in range(0, len(item_comb)):
        # Define Groups
        groupA = dataframe[dataframe[group] == item_comb[i][0]][target]
        groupB = dataframe[dataframe[group] == item_comb[i][1]][target]
        
        # Assumption: Normality
        ntA = shapiro(groupA)[1] < 0.05
        ntB = shapiro(groupB)[1] < 0.05
        # H0: Distribution is Normal! - False
        # H1: Distribution is not Normal! - True
        
        if (ntA == False) & (ntB == False): # "H0: Normal Distribution"
            # Parametric Test
            # Assumption: Homogeneity of variances
            leveneTest = stats.levene(groupA, groupB)[1] < 0.05
            # H0: Homogeneity: False
            # H1: Heterogeneous: True
            if leveneTest == False:
                # Homogeneity
                ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
                # H0: M1 = M2 - False
                # H1: M1 != M2 - True
            else:
                # Heterogeneous
                ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
                # H0: M1 = M2 - False
                # H1: M1 != M2 - True
        else:
            # Non-Parametric Test
            ttest = stats.mannwhitneyu(groupA, groupB)[1] 
            # H0: M1 = M2 - False
            # H1: M1 != M2 - True
            
        temp = pd.DataFrame({"Compare Two Groups":[ttest < 0.05], 
                             "p-value":[ttest],
                             "GroupA_Mean":[groupA.mean()], "GroupB_Mean":[groupB.mean()],
                             "GroupA_Median":[groupA.median()], "GroupB_Median":[groupB.median()],
                             "GroupA_Count":[groupA.count()], "GroupB_Count":[groupB.count()]
                            }, index = [item_comb[i]])
        temp["Compare Two Groups"] = np.where(temp["Compare Two Groups"] == True, "Different Groups", "Similar Groups")
        temp["TestType"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
        
        AB = pd.concat([AB, temp[["TestType", "Compare Two Groups", "p-value","GroupA_Median", "GroupB_Median","GroupA_Mean", "GroupB_Mean",
                                 "GroupA_Count", "GroupB_Count"]]])
        
    with pd.ExcelWriter(f"artifacts/features/similarity_test_{group}-{target}.xlsx", engine='xlsxwriter') as writer:
        AB.to_excel(writer, sheet_name='Similarity Testing', index=False)

    return AB



def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.isocalendar().week
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df["quarter"] = df.date.dt.quarter
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype(int)
    df['is_year_start'] = df.date.dt.is_year_start.astype(int)
    df['is_year_end'] = df.date.dt.is_year_end.astype(int)
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = np.where(df.month.isin([9, 10, 11]), 3, df["season"])
    return df

def rolling_summary_features(df):
    for i in [91, 98, 105, 112, 119, 126, 186, 200, 210, 250, 300, 365, 546, 700]:
        df["sales_roll_mean_"+str(i)]=df.groupby(["store", "item"]).sales.rolling(i).mean().shift(1).values
    return df

def similarity_features(train,df):
    # Store Based
    storesales = train.groupby(["date", "store"]).sales.sum().reset_index()
    ctg_ss = CompareTwoGroups(storesales, group="store", target="sales")
    del storesales

    df["StoreSalesSimilarity"] = np.where(df.store.isin([3,10]), 1, 0)
    df["StoreSalesSimilarity"] = np.where(df.store.isin([4,9]), 2, df["StoreSalesSimilarity"])
    df["StoreSalesSimilarity"] = np.where(df.store.isin([5,6]), 3, df["StoreSalesSimilarity"])

    # Item Based

    itemsales = train.groupby(["date", "item"]).sales.sum().reset_index()
    ctg_is = CompareTwoGroups(itemsales, group = "item", target = "sales")
    del itemsales

    df["ItemSalesSimilarity"] = np.where(df.item.isin([1,4,27,41,47]), 1, 0)
    df["ItemSalesSimilarity"] = np.where(df.item.isin([2,6,7,14,31,46]), 2, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([3,42]), 3, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([8,36]), 4, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([9,43,48]), 5, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([11,12,29,33]), 6, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([13,18]), 7, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([15,28]), 8, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([16,34]), 9, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([19,21,30]), 10, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([20,26]), 11, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([22,25,38,45]), 12, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([23,37,40,44,49]), 13, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([24,35,50]), 14, df["ItemSalesSimilarity"])
    df["ItemSalesSimilarity"] = np.where(df.item.isin([32,39]), 15, df["ItemSalesSimilarity"])
    return df


def lag_features(dataframe, lags, groups = ["store", "item"], target = "sales", prefix = ''):
    dataframe = dataframe.copy()
    for lag in lags:
        dataframe[prefix + str(lag)] = dataframe.groupby(groups)[target].transform(
            lambda x: x.shift(lag))
    return dataframe

def drop_cor(dataframe, name, index):
    ind = dataframe[dataframe.columns[dataframe.columns.str.contains(name)].tolist()+["sales"]].corr().sales.sort_values(ascending = False).index[1:index]
    ind = dataframe.drop(ind, axis = 1).columns[dataframe.drop(ind, axis = 1).columns.str.contains(name)]
    dataframe.drop(ind, axis = 1, inplace = True)

def store_item_features(df):
    df["monthyear"] = df.date.dt.to_period('M')

    for i in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]:
        last_months = df.groupby(["store", "item", "monthyear"]).sales.agg([
            "sum", "mean", "std", "min", "max"]).shift(i).reset_index()
        last_months.columns = ['store', 'item', 'monthyear', 'last_'+str(i)+'months_sales_sum',
                            'last_'+str(i)+'months_sales_mean', 'last_'+str(i)+'months_sales_std',
                            'last_'+str(i)+'months_sales_min', 'last_'+str(i)+'months_sales_max']
        df = pd.merge(df, last_months, how   = "left", on = ["store", "item", "monthyear"])
    drop_cor(df, "last_", 15)
    del last_months, i
    return df

def store_item_similarity_features(df):
    for i in [3, 6, 9, 12]:
        last_months = df.groupby(["store", "monthyear"]).sales.agg([
            "sum", "mean", "std", "min", "max"]).shift(i).reset_index()
        last_months.columns = ['store', 'monthyear', 'store_last_'+str(i)+'months_sales_sum',
                            'store_last_'+str(i)+'months_sales_mean', 'store_last_'+str(i)+'months_sales_std',
                            'store_last_'+str(i)+'months_sales_min', 'store_last_'+str(i)+'months_sales_max']
        df = pd.merge(df, last_months, how = "left", on = ["store", "monthyear"])
    del last_months, i

    # Item Based
    for i in [3, 6, 9, 12]:
        last_months = df.groupby(["item", "monthyear"]).sales.agg([
            "sum", "mean", "std", "min", "max"]).shift(i).reset_index()
        last_months.columns = ['item', 'monthyear', 'item_last_'+str(i)+'months_sales_sum',
                            'item_last_'+str(i)+'months_sales_mean', 'item_last_'+str(i)+'months_sales_std',
                            'item_last_'+str(i)+'months_sales_min', 'item_last_'+str(i)+'months_sales_max']
        df = pd.merge(df, last_months, how = "left", on = ["item", "monthyear"])
    del last_months, i

    # Similarity Based


    for i in [3, 6, 9, 12]:
        last_months = df.groupby(["StoreSalesSimilarity", "monthyear"]).sales.agg([
            "sum", "mean", "std", "min", "max"]).shift(i).reset_index()
        last_months.columns = ['StoreSalesSimilarity', 'monthyear', 'storesim_last_'+str(i)+'months_sales_sum',
                            'storesim_last_'+str(i)+'months_sales_mean', 'storesim_last_'+str(i)+'months_sales_std',
                            'storesim_last_'+str(i)+'months_sales_min', 'storesim_last_'+str(i)+'months_sales_max']
        df = pd.merge(df, last_months, how = "left", on = ["StoreSalesSimilarity", "monthyear"])
    del last_months, i


    for i in [3, 6, 9, 12]:
        last_months = df.groupby(["ItemSalesSimilarity", "monthyear"]).sales.agg([
            "sum", "mean", "std", "min", "max"]).shift(i).reset_index()
        last_months.columns = ['ItemSalesSimilarity', 'monthyear', 'itemsim_last_'+str(i)+'months_sales_sum',
                            'itemsim_last_'+str(i)+'months_sales_mean', 'itemsim_last_'+str(i)+'months_sales_std',
                            'itemsim_last_'+str(i)+'months_sales_min', 'itemsim_last_'+str(i)+'months_sales_max']
        df = pd.merge(df, last_months, how = "left", on = ["ItemSalesSimilarity", "monthyear"])
    del last_months, i

    df.drop("monthyear", axis = 1, inplace = True)
    return df

def day_of_week_features(df):

    df.sort_values(["store", "item", "day_of_week", "date"], inplace = True)
    df = lag_features(df, lags = np.arange(12,41, 1).tolist()+[91, 92, 95, 98, 99, 100, 105, 112, 119, 126, 133, 140, 200, 205, 210, 215, 220, 250],
                    groups = ["store", "item", "day_of_week"], target = 'sales', prefix = 'dayofweek_sales_lag_')
    df[df.columns[df.columns.str.contains("dayofweek_sales_lag_")].tolist()+["sales"]].corr().sales.sort_values(ascending = False)
    drop_cor(df, "dayofweek_sales_lag_", 16)
    df.sort_values(["store", "item", "date"], inplace = True)
    return df

def ewm_features(dataframe, alphas=[0.95, 0.9, 0.8, 0.7, 0.5], lags=[91, 98, 105, 112, 180, 270, 365, 546, 728]):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

def day_of_year_features(df):
    df.sort_values(["day_of_year", "store", "item"], inplace = True)
    df = lag_features(df, lags = [1,2,3,4],
                    groups = ["day_of_year", "store", "item"], target = 'sales', prefix = 'dayofyear_sales_lag_')
    return df
    

def pd_cut(df):
    clus = df.groupby(["store"]).sales.mean().reset_index()
    clus["store_cluster"] =  pd.cut(clus.sales, bins = 4, labels = range(1,5))
    clus.drop("sales", axis = 1, inplace = True)
    df = pd.merge(df, clus, how = "left")
    clus = df.groupby(["item"]).sales.mean().reset_index()
    clus["item_cluster"] =  pd.cut(clus.sales, bins = 5, labels = range(1,6))
    clus.drop("sales", axis = 1, inplace = True)
    df = pd.merge(df, clus, how = "left")
    del clus
    return df


def feature_selection(train,df):
    storesales = train.groupby(["date", "store"]).sales.sum().reset_index().set_index("date")
    CompareTwoGroups(storesales, group = "store", target = "sales")
    itemsales = train.groupby(["date", "item"]).sales.sum().reset_index().set_index("date")
    ctg_is = CompareTwoGroups(itemsales, group = "item", target = "sales")
    ctg_is[ctg_is["Compare Two Groups"] == "Similar Groups"]

    df = create_date_features(df)
    df = rolling_summary_features(df)
    df = similarity_features(train,df)
    df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)
    df = lag_features(df, lags = [91, 92,93,94,95,96, 97, 98, 100, 105, 112, 119, 126, 150,
                              182,200,220, 250, 300, 350, 355, 360,361,362,363, 364,
                              365, 370, 375,380, 546, 600, 650, 680, 690, 700, 710, 728,
                              730, 800, 900, 950, 990, 1000, 1050, 1090, 1095],
                  groups = ["store", "item"], target = 'sales', prefix = 'sales_lag_')
    
    drop_cor(df, "sales_lag", 16)
    df = store_item_features(df)
    df = store_item_similarity_features(df)
    df = day_of_week_features(df)
    df = ewm_features(df)
    df = day_of_year_features(df)
    df = pd_cut(df)
    return df




