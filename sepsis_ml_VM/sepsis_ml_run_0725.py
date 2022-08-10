import sepsis_ml_0725 as ml
import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime, timedelta
import requests
from requests.auth import HTTPBasicAuth
import json


def connect(url, username, password):
    response = requests.get(url,auth=HTTPBasicAuth(username, password), verify=True)
    data = json.loads(response.content)
    #print(data["hasMore"])

    hasMore = data["hasMore"]
    offset = data["offset"]
    #print(offset)
    df = pd.DataFrame(data['items'])
   
    while hasMore:
        querystring = {"offset": offset+ 1000}
        response = requests.get(url,auth=HTTPBasicAuth(username, password), verify=True, params = querystring)
        data = json.loads(response.content)

        hasMore = data["hasMore"]
        offset = data["offset"]
        #print(hasMore)
        #print(offset)
        df = pd.concat([df,pd.DataFrame(data['items'])])

    return df


def upload(url, username, password, output):
    response = requests.post(url,headers={"Content-Type":"application/json"},auth=HTTPBasicAuth(username, password), data = output)


def run(test_data):
    
    dtypes = {"pat_id": int, "csn": int, "HR": float, "O2Sat": float, "Temp": float, 
             "SBP": float, "DBP": float, "Resp": float, "EtCO2": float, 
             "AST": float, "Alkalinephos": float, "BUN": float, "BaseExcess": float,
             "Bilirubin_total": float, "Calcium": float, "Creatinine": float,
             "FiO2": float, "Glucose": float, "HCO3": float, "Hct": float, 
             "Hgb": float, "Lactate": float, "Magnesium": float, "PTT": float,
             "PaCO2": float, "PaO2": float, "Phosphate": float, "Platelets": float,
             "Potassium": float, "SaO2": float, "Sodium": float, "WBC": float,
             "pH": float, "gcs_total_score": float, "age": float, "gender": int}
    
    #test_data = pd.read_csv(test_data_path, dtype = dtypes, parse_dates= ["recorded_time", "hospital_admission_date_time"])
    
    thresh = vitals_thresh = {"HR": (0,250),
                 "O2Sat": (0,100),
                 "Temp": (25,45),
                 "SBP": (0,260),
                 "DBP": (0, 220),
                 "MAP": (0,260),
                 'Resp': (0,80),
                 'EtCO2': (0, 60),
                  "pH": (6.7, 8),
                  "PaCO2": (15, 150),
                  "SaO2": (0,100),
                  "AST": (0, 10000),
                  "BUN": (0,200),
                  "Alkalinephos": (0, 10000),
                  "Calcium": (0,20),
                  "Chloride": (60,150),
                  "Creatinine": (0, 15),
                  "Glucose": (0, 1200),
                  "Lactate": (0,20),
                  "Magnesium": (0,10), 
                  "Phosphate": (0,20),
                  "Potassium": (0,10),
                  "Bilirubin_total": (0,30),
                  "Hct": (0, 75),
                   "Hgb": (0,25),
                   "PTT": (0,150),
                   "WBC": (0,150),
                   "Platelets": (0,1000)}

    test_data = ml.data_clean(test_data, thresh)
    
    variables = ['HR',
                 'O2Sat',
                 'Temp',
                 'SBP',
                 'MAP',
                 'DBP',
                 'Resp',
                 'EtCO2',
                 'AST',
                 'Alkalinephos',
                 'BUN',
                 'BaseExcess',
                 'Bilirubin_total',
                 'Calcium',
                 'Chloride',
                 'Creatinine',
                 'FiO2',
                 'Glucose',
                 'HCO3',
                 'Hct',
                 'Hgb',
                 'Lactate',
                 'Magnesium',
                 'PTT',
                 'PaCO2',
                 'PaO2',
                 'Phosphate',
                 'Platelets',
                 'Potassium',
                 'SaO2',
                 'Sodium',
                 'WBC',
                 'pH',
                 'gcs_total_score']
    
    demographics = ["pat_id", "csn", "age", "gender"]
    
    stat = test_data[demographics].drop_duplicates()
    
    test_data = test_data.drop(["age", "gender"], axis = 1)
    
    test_data = ml.resampling(test_data)
    test_data = test_data.groupby(["pat_id", "csn"]).apply(lambda v: ml.rolling_overlap(v, 6, variables, 3))
    test_data = test_data.drop(["pat_id", "csn"], axis = 1).reset_index(drop = False).rename(columns = {"level_2" : "LOS"})
    test_data = test_data.merge(stat, on = ["pat_id", "csn"], how = "left")
  
    return test_data


def select_process_data(test_data, current_time):
    
    # 1) filter only encounters within 48 hour from current time
    final_test = test_data.copy()
    
    final_test.loc[:,"keep"] = (current_time - final_test["hospital_admission_date_time"]).dt.total_seconds()/3600
    keep_csns_filter1 = final_test.loc[final_test["keep"] <= 49].csn.unique()
    final_test = final_test[final_test["csn"].isin(keep_csns_filter1)]
    
    # 2) filter data within 0-48 hour since admission
    final_test["keep"] = (final_test["recorded_time"] - final_test["hospital_admission_date_time"]).dt.total_seconds()/3600.0
    final_test["keep"] = (final_test["keep"] <=49) & (final_test["keep"] >= 0)
    final_test = final_test.copy()
    final_test = final_test.loc[final_test.keep, :].reset_index(drop = True)
    final_test.drop(["keep"], axis = 1, inplace = True)
    
    return final_test
    
def get_hist_data(test_data, current_time):
    hist_data = test_data.copy()
    hist_data["past_time"] = (current_time - hist_data["recorded_time"]).dt.total_seconds()/3600
   
    g = ['AST', 'Alkalinephos', 'BUN',
       'Bilirubin_total', 'Calcium', 'Chloride', 'Creatinine', 'EtCO2',
       'Glucose', 'HR', 'Hct', 'Hgb', 'Magnesium', 'O2Sat', 'Phosphate',
       'Platelets', 'Potassium', 'Resp', 'Sodium', 'Temp', 'WBC',
       'gcs_total_score', 'PTT', 'BaseExcess', 'HCO3', 'PaCO2', 'Lactate', 'PaO2', 'SaO2', 'pH',
       'FiO2', 'SBP', 'DBP', 'MAP']
    
    hist_data[g] = hist_data[g].notnull().astype('int')

    t = hist_data["past_time"].values
    t = np.vstack([t]*len(g))

    hist_data[g] = np.where(hist_data[g] == 1, np.transpose(t), hist_data[g])

    hist_data.replace(0, np.nan, inplace = True)
    hist_data[g] = hist_data.groupby(by = ["csn", "pat_id"])[g].fillna(method = 'ffill')
    hist_data.drop_duplicates(subset = ["csn", "pat_id"], keep = "last", inplace = True)
    add_cols = ['HR_interval_f1', 'HR_interval_f2', 'HR_diff', 'O2Sat_interval_f1', 'O2Sat_interval_f2', 'O2Sat_diff', 'Temp_interval_f1', 'Temp_interval_f2', 'Temp_diff', 'SBP_interval_f1', 'SBP_interval_f2', 'SBP_diff', 'MAP_interval_f1', 'MAP_interval_f2', 'MAP_diff', 'DBP_interval_f1', 'DBP_interval_f2', 'DBP_diff', 'Resp_interval_f1', 'Resp_interval_f2', 'Resp_diff', 'EtCO2_interval_f1', 'EtCO2_interval_f2', 'EtCO2_diff', 'BaseExcess_interval_f1', 'BaseExcess_interval_f2', 'BaseExcess_diff', 'HCO3_interval_f1', 'HCO3_interval_f2', 'HCO3_diff', 'FiO2_interval_f1', 'FiO2_interval_f2', 'FiO2_diff', 'pH_interval_f1', 'pH_interval_f2', 'pH_diff', 'PaCO2_interval_f1', 'PaCO2_interval_f2', 'PaCO2_diff', 'SaO2_interval_f1', 'SaO2_interval_f2', 'SaO2_diff', 'AST_interval_f1', 'AST_interval_f2', 'AST_diff', 'BUN_interval_f1', 'BUN_interval_f2', 'BUN_diff', 'Alkalinephos_interval_f1', 'Alkalinephos_interval_f2', 'Alkalinephos_diff', 'Calcium_interval_f1', 'Calcium_interval_f2', 'Calcium_diff', 'Chloride_interval_f1', 'Chloride_interval_f2', 'Chloride_diff', 'Creatinine_interval_f1', 'Creatinine_interval_f2', 'Creatinine_diff', 'Glucose_interval_f1', 'Glucose_interval_f2', 'Glucose_diff', 'Lactate_interval_f1', 'Lactate_interval_f2', 'Lactate_diff', 'Magnesium_interval_f1', 'Magnesium_interval_f2', 'Magnesium_diff', 'Phosphate_interval_f1', 'Phosphate_interval_f2', 'Phosphate_diff', 'Potassium_interval_f1', 'Potassium_interval_f2', 'Potassium_diff', 'Bilirubin_total_interval_f1', 'Bilirubin_total_interval_f2', 'Bilirubin_total_diff', 'Hct_interval_f1', 'Hct_interval_f2', 'Hct_diff', 'Hgb_interval_f1', 'Hgb_interval_f2', 'Hgb_diff', 'PTT_interval_f1', 'PTT_interval_f2', 'PTT_diff', 'WBC_interval_f1', 'WBC_interval_f2', 'WBC_diff', 'Platelets_interval_f1', 'Platelets_interval_f2', 'Platelets_diff', 'HR_mean', 'O2Sat_mean', 'SBP_mean', 'MAP_mean', 'Resp_mean', 'HR_median', 'O2Sat_median', 'SBP_median', 'MAP_median', 'Resp_median', 'HR_min', 'O2Sat_min', 'SBP_min', 'MAP_min', 'Resp_min', 'HR_max', 'O2Sat_max', 'SBP_max', 'MAP_max', 'Resp_max', 'HR_std', 'O2Sat_std', 'SBP_std', 'MAP_std', 'Resp_std', 'HR_dstd', 'O2Sat_dstd', 'SBP_dstd', 'MAP_dstd', 'Resp_dstd', 'HR_score', 'Temp_score', 'Resp_score', 'MAP_score', 'Creatinine_score', 'qsofa', 'Platelets_score', 'Bilirubin_score']

    nan_df = pd.DataFrame(columns = add_cols)

    hist_data = pd.concat([hist_data, nan_df])
    hist_data.iloc[:,3:] = round(hist_data.iloc[:,3:],3)

    return hist_data

    

if __name__ == "__main__":

    # get user input parameters
    parser = argparse.ArgumentParser("real-time sepsis model run")
    parser.add_argument("model_dir", help = "Directory to the pre-trained models.", type=str)
    args = parser.parse_args()
    
    # get current time
    current_time = pd.to_datetime(str(pd.Timestamp.now())[:-7])
    print(current_time)
    
    # credidentials for API
    username = 'Sepsis_ML'
    password = 'jfVDS756F$jkf&@*'
    
    
    # receive input data with GET
    url = 'https://prd-rta-app01.eushc.org:8443/ords/rta/sepsisml/derivedcache'
    test_data = connect(url, username, password)
    
    print("completed reading")

    ### FIRST DATA PROCESSING ###
    # convert names
    sepsis_ml_map = pd.read_csv("sepsis_ml_map.csv")
    sepsis_ml_map = sepsis_ml_map[["Category", "event_cd"]]
    
    test_data = test_data.rename(columns = {"person_id": "pat_id", "encntr_id": "csn",
                                            "gender_disp": "gender",
                                            "arrive_dt_tm": "hospital_admission_date_time",
                                            "event_start_dt_tm": "recorded_time"})

    # convert days to ETC but w/o time zone
    for col in ["recorded_time", "hospital_admission_date_time", "birth_dt_tm"]:
        test_data[col] = pd.to_datetime(test_data[col]).dt.tz_convert('US/Eastern')
        test_data[col] = test_data[col].dt.tz_localize(None)
    
    
    
    
    ### FILTERING ###
    # current time calculation
    
    test_data  = select_process_data(test_data, current_time)

    
    test_data["result_val"] = pd.to_numeric(test_data["result_val"], errors='coerce')
    test_data["pat_id"] = pd.to_numeric(test_data["pat_id"], errors='coerce')
    test_data["csn"] = pd.to_numeric(test_data["csn"], errors='coerce')
    test_data["age"] = np.floor((test_data["hospital_admission_date_time"] - test_data["birth_dt_tm"]).dt.total_seconds() / (60 * 60 * 24 * 365))


    ### STATIC DATA EXTRACTION ###
    static = test_data[["csn", "pat_id", "gender", "hospital_admission_date_time","age"]].copy()
    static = static.drop_duplicates()
    static.loc[~(static["gender"].isin(["Female", "Male"])), "gender"] = np.nan
    static["gender"] = static["gender"].replace({"Female": 0, "Male" : 1}).fillna(-1)

    
    
    ### LONGITUDINAL DATA EXTRACTION ###
    test_data = test_data.merge(sepsis_ml_map, on = "event_cd", how = "left")
    test_data = test_data.pivot_table(index=['csn', "pat_id", "recorded_time"], columns='Category', values='result_val')
    test_data = test_data.reset_index()
    
    
    ### TEST_DATA CREATE ###
    test_data = test_data.merge(static, on = ["csn", "pat_id"], how = "left")
    
    for c in list(sepsis_ml_map.Category.unique()):
        if not (c in test_data.columns): 
            test_data[c] = np.nan
            
            
    ### THIRD DATA PREPROCESSING ###
    test_data["SBP"] = test_data["SBP"].fillna(test_data["SBP_Cuff"])
    test_data["MAP"] = test_data["MAP"].fillna(test_data["MAP_Cuff"])
    test_data["DBP"] = test_data["DBP"].fillna(test_data["DBP_Cuff"])
    
    test_data = test_data.drop(["SBP_Cuff", "MAP_Cuff", "DBP_Cuff", "Fibrinogen"], axis = 1)
    
    
    ### SAVE HISTORICAL DATA ###
    hist_data = get_hist_data(test_data).reset_index(drop = True)

    
    ### RUN MODEL PREPROCESSING ###
    
    print(len(test_data))
    
    test_data = run(test_data)
    test_data = test_data.rename(columns = {"LOS": "los"})
    
    print("completed preprocessing")
    test_data.to_csv("test_data_saved_debug.csv", index = False)

    
    ### read output and get historical data for future use ###
   
    url_output = 'https://prd-rta-app01.eushc.org:8443/ords/rta/sepsisml/outputcache'
    historical_output = connect(url_output, username, password)
    
    historical_output = historical_output.sort_values(by = "run_date", ascending = True)
    historical_output = historical_output[["csn", "pat_id", "predictedprobability", "run_date"]]

    historical_output = historical_output.dropna().drop_duplicates(subset = ["csn", "pat_id"], keep = "last")
    historical_output = historical_output.rename(columns = {"predictedprobability": "PastProbability", "run_date": "PastRunDate"})

    historical_output["PastRunDate"] = pd.to_datetime(historical_output["PastRunDate"])
    historical_output["PastRunDate"] = historical_output["PastRunDate"].dt.tz_convert('US/Eastern')
    historical_output["PastRunDate"] = historical_output["PastRunDate"].dt.tz_localize(None)
    historical_output["pat_id"] = pd.to_numeric(historical_output["pat_id"], errors='coerce')
    historical_output["csn"] = pd.to_numeric(historical_output["csn"], errors='coerce')
    historical_output["PastProbability"] = pd.to_numeric(historical_output["PastProbability"], errors='coerce')
  
    ### TESTING ###
    
    test_data = ml.preprocess(test_data)
    
    test_data = test_data.sort_values("rel_time", ascending = True).drop_duplicates(subset = ["csn", "pat_id"], keep = "last")
    
    
    test_set = list(test_data.csn.unique())
    model_path = args.model_dir
    
    current_time_formatted = str(current_time.tz_localize("US/Eastern"))

    result = ml.predict(test_set, test_data, model_path, 0.48, hist_data, current_time_formatted)
    result = result.rename(columns = {"LOS": "los"})
    #result = pd.concat([result, keep_data])
    
    print("completed model testing")

  
    result = result.merge(historical_output, on = ["csn", "pat_id"], how = "left")

    result["prob_diff"] = round((result["PredictedProbability"] - result["PastProbability"])*100, 2)
    result["PreviousAlert"] = round(result["PredictedProbability"] * 100, 2).astype(str) + '% ' + (result["prob_diff"]).map('({0:+}%)'.format)
    
    
    result["run_date_relative"] = result["rel_time"]
    
    result["run_date"] = current_time
    result["past_curr_diff"] = (result["run_date"] - result["PastRunDate"]).dt.total_seconds()/3600
    result["PastRunDate"] = (result["PastRunDate"].dt.tz_localize("US/Eastern")).astype(str)
    result["run_date"] = current_time_formatted
                                       
    result["PriorAlertTime"] = result["PastRunDate"] + " " + (result["past_curr_diff"]).map('({:,.2f} hrs ago)'.format)
    result.drop(["past_curr_diff"], axis = 1)
    
    result.to_csv("output_debug.csv", index = False)
    
    ### UPLOAD OUTPUT TO DATABASE ###
    ### DEBUG PURPOSE ###
    
    result = result.fillna("nan")
    
    output = result.to_dict(orient='records')
    
    url_output = 'https://prd-rta-app01.eushc.org:8443/ords/rta/sepsisml/outputcache'
    
    for i_output in output:
        upload(url_output, username, password, json.dumps(i_output))
    
   
    ### LOGGING ###
    
    logname = "sepsis_ml_execution.log"
    logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

    logging.info("Result Outputed")

    print("result outputed")
    
     

