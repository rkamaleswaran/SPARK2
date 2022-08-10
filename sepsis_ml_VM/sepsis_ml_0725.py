import numpy as np
import matplotlib.pyplot as plt
from requests.auth import HTTPBasicAuth
import requests
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from hyperopt import STATUS_OK, hp, fmin, tpe
from tqdm import tqdm
import base64
import json
import shap
import numpy as np, os, sys

def upload(url, username, password, output):
    response = requests.post(url,auth=HTTPBasicAuth(username, password), verify=True, data = output, headers = {"Content-Type": "application/json"})


##### CLEANING ALGORITHMS #####

sep_index = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
             'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
             'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets']
con_index = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

demographics = ["age", "gender"]


def resampling(new_merged):
    # random origin
    origin = pd.to_datetime("2000-01-01")

    # sort index
    new_merged["rel_time"] = (new_merged["recorded_time"] - new_merged["hospital_admission_date_time"]) + origin
    new_merged = new_merged.set_index("rel_time").sort_index()

    # aggregate each hour using median value

    df = new_merged.groupby(by = ["pat_id", "csn"]).resample("1H", label = "right", origin = origin).median()
    
    df = df.drop(["csn", "pat_id"],axis=1)
    df = df.reset_index()
    df["rel_time"] = (df["rel_time"] - origin).dt.total_seconds() / (60 * 60)
    
    return df

def data_clean(chunk, thresholds):
    process_cols = list(thresholds.keys())
    for feature in process_cols:
        chunk.loc[:,feature] = chunk[feature].replace(r'\>|\<|\%|\/|\s','',regex=True)
        chunk.loc[:,feature] = pd.to_numeric(chunk[feature], errors='coerce')
        mask_ind = (chunk[feature] < thresholds[feature][1]) & (chunk[feature] > thresholds[feature][0])
        chunk.loc[~mask_ind, feature]  = np.nan
    return chunk




def rolling_overlap(temp, window, variables, overlap):
    rolled= temp.copy()
    rolled[variables] = rolled.rolling(window, min_periods = 1)[variables].aggregate("median")
    #rolled[bed_var] = rolled.rolling(window, min_periods = 1)[bed_var].aggregate("max")
    rolled = rolled.reset_index(drop = True)
    start = window - 1
    return rolled.iloc[0::overlap]

###### DERIVED FEATURES #######


def feature_informative_missingness(patient, sep_columns = con_index + sep_index):
    
    for sep_column in con_index + sep_index:
        
        nonmissing_idx = patient.index[~patient[sep_column].isna()].tolist()
        f1_name = sep_column + "_interval_f1"
        f2_name = sep_column + "_interval_f2"
        diff_name = sep_column + "_diff"

        patient.loc[nonmissing_idx,f1_name] = np.arange(1,len(nonmissing_idx)+1)
        patient[f1_name] = patient[f1_name].ffill().fillna(0)

        v = (0+patient[sep_column].isna()).replace(0,np.nan)
        cumsum = v.cumsum().fillna(method='pad')
        reset = -cumsum[v.isnull()].diff().fillna(cumsum)
        patient[f2_name] = v.where(v.notnull(), reset).cumsum().fillna(0)
        
        if nonmissing_idx==[]:
            patient.loc[:, f2_name] = -1
        else:
            patient.loc[:nonmissing_idx[0]-1, f2_name] = -1
        
        patient[diff_name] = patient.loc[nonmissing_idx, sep_column].diff()
        patient[diff_name] = patient[diff_name].fillna(method = "ffill")    
            
        
    return patient

def feature_slide_window(vitals):

    diff = vitals.shift(-1) - vitals
    rolling_mean = vitals.copy()
    rolling_mean = rolling_mean.groupby("csn").rolling(6, min_periods = 1).mean().reset_index(drop = True)
    rolling_mean = rolling_mean.rename(columns = {'HR': "HR_mean", 'O2Sat': "O2Sat_mean", 'SBP':"SBP_mean", 
                                                  'MAP': "MAP_mean", 'Resp': "Resp_mean"})
    rolling_median = vitals.groupby("csn").rolling(6, min_periods = 1).median().reset_index(drop = True)
    rolling_median = rolling_median.rename(columns = {'HR': "HR_median", 'O2Sat': "O2Sat_median", 'SBP':"SBP_median", 
                                                  'MAP': "MAP_median", 'Resp': "Resp_median"})
    rolling_min = vitals.groupby("csn").rolling(6, min_periods = 1).min().reset_index(drop = True)
    rolling_min = rolling_min.rename(columns = {'HR': "HR_min", 'O2Sat': "O2Sat_min", 'SBP':"SBP_min", 
                                                  'MAP': "MAP_min", 'Resp': "Resp_min"})

    rolling_max = vitals.groupby("csn").rolling(6, min_periods = 1).max().reset_index(drop = True)
    rolling_max = rolling_max.rename(columns = {'HR': "HR_max", 'O2Sat': "O2Sat_max", 'SBP':"SBP_max", 
                                                  'MAP': "MAP_max", 'Resp': "Resp_max"})

    rolling_std = vitals.groupby("csn").rolling(6, min_periods = 1).std().reset_index(drop = True)
    rolling_std = rolling_std.rename(columns = {'HR': "HR_std", 'O2Sat': "O2Sat_std", 'SBP':"SBP_std", 
                                                  'MAP': "MAP_std", 'Resp': "Resp_std"})

    rolling_diff_std = diff.groupby("csn").rolling(6, min_periods = 1).std().reset_index(drop = True)
    rolling_diff_std = rolling_diff_std.rename(columns = {'HR': "HR_dstd", 'O2Sat': "O2Sat_dstd", 'SBP':"SBP_dstd", 
                                                  'MAP': "MAP_dstd", 'Resp': "Resp_dstd"})

    rolling_vitals = pd.concat([rolling_mean, rolling_median, rolling_min, rolling_max, rolling_std, rolling_diff_std], axis = 1)
    
    
    return rolling_vitals

def feature_empiric_score(temp):
    
    
    # HEART RATE SCORING
    temp["HR_score"] = 0
    mask = (temp["HR"] <= 40) | (temp["HR"] >= 131)
    temp.loc[mask,"HR_score"] = 3
    mask = (temp["HR"] <= 130) & (temp["HR"] >= 111)
    temp.loc[mask,"HR_score"] = 2
    mask = ((temp["HR"] <= 50) & (temp["HR"] >= 41)) | ((temp["HR"] <= 110) & (temp["HR"] >= 91))
    temp.loc[mask,"HR_score"] = 1
    temp.loc[temp["HR"].isna(),"HR_score"] = np.nan


    # TEMPERATURE SCORING

    temp["Temp_score"] = 0

    mask = (temp["Temp"] <= 35)
    temp.loc[mask,"Temp_score"] = 3
    mask = (temp["Temp"] >= 39.1)
    temp.loc[mask,"Temp_score"] = 2
    mask = ((temp["Temp"] <= 36.0) & (temp["Temp"] >= 35.1)) | ((temp["Temp"] <= 39.0) & (temp["HR"] >= 38.1))
    temp.loc[mask,"Temp_score"] = 1

    temp.loc[temp["Temp"].isna(),"Temp_score"] = np.nan


    # Resp Score

    temp["Resp_score"] = 0

    mask = (temp["Resp"] < 8) | (temp["Resp"] > 25)
    temp.loc[mask,"Resp_score"] = 3
    mask = ((temp["Resp"] <= 24) & (temp["Resp"] >= 21))
    temp.loc[mask,"Resp_score"] = 2
    mask = ((temp["Resp"] <=11) & (temp["Resp"] >= 9))
    temp.loc[mask,"Resp_score"] = 1

    temp.loc[temp["Resp"].isna(),"Resp_score"] = np.nan

    #MAP Score
    temp["MAP_score"] = 1
    mask = (temp["MAP"] >= 70)
    temp.loc[mask, "MAP_score"] = 0
    temp.loc[temp["MAP"].isna(),"MAP_score"] = np.nan
    
    # Creatinine score:

    temp["Creatinine_score"] = 3

    mask = (temp["Creatinine"] < 3.5)
    temp.loc[mask, "Creatinine_score"] = 2
    mask = (temp["Creatinine"] < 2)
    temp.loc[mask, "Creatinine_score"] = 1
    mask = (temp["Creatinine"] < 1.2)
    temp.loc[mask, "Creatinine_score"] = 0
    temp.loc[temp["Creatinine"].isna(),"Creatinine_score"] = np.nan


    # qsofa:
    temp["qsofa"] = 0
    mask = (temp["SBP"] <= 100) & (temp["Resp"] >= 22)
    temp.loc[mask, "qsofa"] = 1
    mask = (temp["SBP"].isna()) | (temp["Resp"].isna())
    temp.loc[mask, "qsofa"] = np.nan

    # Platelets score:
    temp["Platelets_score"] = 0
    mask = (temp["Platelets"] <= 150)
    temp.loc[mask, "Platelets_score"] = 1
    mask = (temp["Platelets"] <= 100)
    temp.loc[mask, "Platelets_score"] = 2
    mask = (temp["Platelets"] <= 50)
    temp.loc[mask, "Platelets_score"] = 3

    temp.loc[temp["Platelets"].isna(),"Platelets_score"] = np.nan



    # Bilirubin score:
    temp["Bilirubin_score"] = 3
    mask = (temp["Bilirubin_total"] < 6)
    temp.loc[mask, "Bilirubin_score"] = 2
    mask = (temp["Bilirubin_total"] < 2)
    temp.loc[mask, "Bilirubin_score"] = 1
    mask = (temp["Bilirubin_total"] < 1.2)
    temp.loc[mask, "Bilirubin_score"] = 0
    temp.loc[temp["Bilirubin_total"].isna(),"Bilirubin_score"] = np.nan
    
    return(temp)


# 62 informative missingness features, 31 differential features and 37 raw variables

def preprocess(df_process):
    
    #print(len(df_process.columns)) 
    print("Extracting informative features")
    
    groups = []
    
    with tqdm(total= df_process.csn.nunique()) as pbar:
        for _, case in df_process.groupby(["csn", "pat_id"]):
            groups.append(feature_informative_missingness(case))             
            pbar.update(1)
            
    temp = pd.concat(groups).reset_index(drop = True)
    
    print("Completed Extracting informative features")
    #print(len(temp.columns))
    temp = temp.fillna(method='ffill').reset_index(drop = True)
    #print(len(temp.columns))
    print("Extracting Rolling features")
    
    vitals = temp.copy()
    vitals = vitals[["csn", 'HR', 'O2Sat', 'SBP', 'MAP', 'Resp']]
    vitals = feature_slide_window(vitals).reset_index(drop = True)
    #print("vitals:", list(vitals.columns))
    #print(len(vitals.columns))
    #vitals = feature_slide_window(vitals).reset_index(drop = True).drop(["csn"], axis = 1)
    print("Completed Extracting Rolling features")
    
    new = pd.concat([temp, vitals], axis = 1)
    #print(len(new.columns))

    # add 8 empiric features scorings
    print("Extracting Score Features")
    
    new = feature_empiric_score(new)
    print("Completed Extracting Score Features")
    print("Preprocessing completed with total of", len(list(new.columns)), "features")
        
    return new



def downsample(x):
    
    pos = x[x["SepsisLabel"] == 1]
    neg = x[x["SepsisLabel"] == 0]
    
    if len(pos) < len(neg):
        neg = neg.sample(n=len(pos), replace = False, random_state = 10002)
        
    new = pos.append(neg)
    new = new.sample(frac = 1, replace = False)
    
    return new


def load_model_predict(X_test, k_fold, path):
    #"ensemble the five XGBoost models by averaging their output probabilities"

    test_pred = np.zeros((X_test.shape[0], k_fold))
    all_shap_values = np.zeros((X_test.shape[0], X_test.shape[1]))
    X_test = xgb.DMatrix(X_test)
    exp = 0  
    for k in range(k_fold):
        model_path_name = path + 'model{}.mdl'.format(k+1)
        xgb_model = xgb.Booster(model_file = model_path_name)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        all_shap_values = all_shap_values + shap_values
        exp = exp + explainer.expected_value
        y_test_pred = xgb_model.predict(X_test)
        test_pred[:, k] = y_test_pred
    
    test_pred = pd.DataFrame(test_pred)
    result_pro = test_pred.mean(axis=1)
    return result_pro, all_shap_values/5, exp/5


def predict(data_set,
            data_dir,
            model_path,
            risk_threshold,
            hist_data,
            current_time):
  
    patient_df = data_dir.copy()
    new_set = patient_df.drop(["csn", "pat_id", "los", "rel_time"], axis = 1).copy()
    features = new_set.values
    feature_names = np.array(new_set.columns)
    
    predict_pro, shaps, exp = load_model_predict(features, k_fold = 5, path = './' + model_path + '/')

    for i in range(0, len(new_set)):
        shap.force_plot(exp, shaps[i,:], new_set.iloc[i,:], matplotlib = True, show = False)
            
        pt_csn = patient_df.iloc[i, 0]
        pt_pat_id = patient_df.iloc[i, 1]
        pt_los = patient_df.iloc[i, 2]
        plt.savefig("temp.png")
        plt.close()

        with open("temp.png", mode = 'rb') as file:
            img = file.read()
            encoded_string = base64.b64encode(img)
            encoded_string = str(encoded_string)[2:-1]

        output = {"csn": str(pt_csn), "pat_id": str(pt_pat_id), "los": int(pt_los), "content_type": "image/png"}
        output["run_date"] = current_time
        output["run_date_relative"] = str(patient_df.iloc[i, 3])
        output["img_content"] = encoded_string
        upload(url = "https://prd-rta-app01.eushc.org:8443/ords/rta/sepsisml/imgcache", username = 'Sepsis_ML', password = 'jfVDS756F$jkf&@*', output = json.dumps(output))
        
        
    ### TOP 10 MOST **POSITIVIE** IMPACT ### -- change if needed
    hist_times = hist_data[list(feature_names)].values
    
    shaps = np.array(shaps)
    sort_index = (-shaps).argsort(axis = 1)
    ranked_shap = shaps[np.arange(len(shaps))[:,None], sort_index]
    
    top_10 = sort_index[:, :10]
    top_10_features = feature_names[top_10]
    
    top_10_hist_times = hist_times[np.arange(len(shaps))[:,None],top_10]

    top_10_str_list = []
    for a,b in zip(top_10_features,top_10_hist_times):
        top_10_str = ''
        for i,j in zip(a,b):
            top_10_str += i+ " (" +str(j) + "), "
        top_10_str_list.append([top_10_str])
  
    
    
    PredictedProbability = np.array(predict_pro)
    PredictedLabel = [0 if i <= risk_threshold else 1 for i in predict_pro]
        
    temp_result = patient_df[["csn", "pat_id", "los", "rel_time"]].copy().reset_index(drop = True)
        
    temp_result["PredictedProbability"] = PredictedProbability
    temp_result["PredictedSepsisLabel"] = PredictedLabel
    temp_result["ranked_shap"] = pd.Series(list(ranked_shap)).astype(str)
    temp_result["shap"] = pd.Series(top_10_str_list)

    return temp_result
        
