# SepsisML 

SepsisML is a ML-based real time sepsis warning system created by the Kamaleswaran Lab. Every three hours, the model outputs patient's probability for developing sepsis. 

# 1. Data extraction

The following variables are extracted from the EHR and saved to https://prd-rta-app01.eushc.org:8443/ords/rta/sepsisml/derivedcache.

| Variable      | Category (LOINC)|
| ----------- | ----------- |
| pat_id      | Demographics       |
| csn   | Demographics        |
| Alkalinephos | Labs (6768-6) |
| AST | Labs (1920-8) |
| BaseExcess | Labs (1925-7) |
| Bilirubin_total | Labs (1975-2) |
| BUN | Labs (3094-0) |
| Calcium | Labs (17861-6) |
| Chloride | Labs (2075-0) |
| Creatinine | Labs (2160-0) |
| FiO2 | Labs (3150-0) |
| Glucose | Labs (2345-7) |
| HCO3 | Labs (1959-6) |
| Hct | Labs (4544-3) |
| Hgb | Labs (718-7) |
| Lactate | Labs (2518-9) |
| Magnesium | Labs (19123-9) |
| PaCO2 | Labs (2019-8) |
| PaO2 | Labs (2703-7) |
| pH | Labs (2744-1) |
| Phsophate | Labs (2777-1) |
| Platelets | Labs (777-3) |
| Potassium | Labs (2823-3) |
| PTT | Labs (14979-9) |
| SaO2 | Labs (2708-6) |
| Sodium | Labs (2951-2) |
| WBC | Labs (6690-2) |
| HR | Vitals |
| O2Sat | Vitals |
| Temp | Vitals |
| Resp | Vitals |
| SBP | Vitals |
| MBP | Vitals |
| DBP | Vitals |
| EtCO2 | Vitals |
| gcs_total_score | Vitals |
| age | Demographics |
| is_female | Demographics |
| hospital_admission_time | Demographics |


**snipped of the API call from sepsis_ml_run_0725.py**
``` python
# API call for GET and outputs dataset in pandas DataFrame

def connect(url, username, password):
    response = requests.get(url,auth=HTTPBasicAuth(username, password), verify=True)
    data = json.loads(response.content)

    hasMore = data["hasMore"]
    offset = data["offset"]

    df = pd.DataFrame(data['items'])
   
    while hasMore:
        querystring = {"offset": offset+ 1000}
        response = requests.get(url,auth=HTTPBasicAuth(username, password), verify=True, params = querystring)
        data = json.loads(response.content)

        hasMore = data["hasMore"]
        offset = data["offset"]
        df = pd.concat([df,pd.DataFrame(data['items'])])

    return df

# receive input data with GET
url = 'https://prd-rta-app01.eushc.org:8443/ords/rta/sepsisml/derivedcache'
test_data = connect(url, username, password)
```


The table https://prd-rta-app01.eushc.org:8443/ords/rta/sepsisml/derivedcache is formatted as following:

| person_id      | encntr_id | gender_disp | arrive_dt_tm | birth_dt_tm | event_cd | display | result_val | valid_from_dt_tm | event_start_dt_tm |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 0000 | 1234| Female| 2000-08-03T23:59:00Z | 1994-04-12T08:00:00Z | 268390711 | MAP, Cuff | 100 | 2022-08-02T15:13:00Z | 2022-08-02T15:00:00Z |



# 2. Preprocessing

Preprocessing of the data must be performed to achieve an hourly dataset with the variables listed above being each columns. i.e.

| pat_id      | csn | los | is_female | hospital_admission_time | age | Alkalinephos | AST | ... | MAP | ...|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 0000 | 1234| 3 | Female| 2000-08-03T23:59:00Z | 27 | NaN | NaN | ... | 100 | ... |

## 2.1 Datatype conversion

```python
for col in ["event_start_dt_tm", "arrive_dt_tm", "birth_dt_tm"]:
    test_data[col] = pd.to_datetime(test_data[col]).dt.tz_convert('US/Eastern')
    test_data[col] = test_data[col].dt.tz_localize(None)

test_data["result_val"] = pd.to_numeric(test_data["result_val"], errors='coerce')
test_data["pat_id"] = pd.to_numeric(test_data["pat_id"], errors='coerce')
test_data["csn"] = pd.to_numeric(test_data["csn"], errors='coerce')
test_data["age"] = np.floor((test_data["hospital_admission_date_time"] - test_data["birth_dt_tm"]).dt.total_seconds() / (60 * 60 * 24 * 365))

```

## 2.2 Filter only 48 hour since admission
```python

def select_process_data(test_data, current_time):
    
    # 1) filter only encounters within 48 hour from current time
    final_test = test_data.copy()
    
    final_test.loc[:,"keep"] = (current_time - final_test["arrive_dt_tm"]).dt.total_seconds()/3600
    keep_csns_filter1 = final_test.loc[final_test["keep"] <= 49].csn.unique()
    final_test = final_test[final_test["csn"].isin(keep_csns_filter1)]
    
    # 2) filter data within 0-48 hour since admission
    final_test["keep"] = (final_test["event_start_dt_tm"] - final_test["arrive_dt_tm"]).dt.total_seconds()/3600.0
    final_test["keep"] = (final_test["keep"] <=49) & (final_test["keep"] >= 0)
    final_test = final_test.copy()
    final_test = final_test.loc[final_test.keep, :].reset_index(drop = True)
    final_test.drop(["keep"], axis = 1, inplace = True)
    
    return final_test

test_data  = select_process_data(test_data, current_time)

```

## 2.3 Rename columns Mapping Labs and Vitals

```python
test_data = test_data.rename(columns = {"person_id": "pat_id",
                                        "encntr_id": "csn",
                                        "gender_disp": "is_female",
                                        "arrive_dt_tm": "hospital_admission_date_time",
                                        "event_start_dt_tm": "recorded_time"})

```

## 2.4 Extract static variables
```python
static = test_data[["csn", "pat_id", "is_female", "hospital_admission_date_time","age"]].copy()
static = static.drop_duplicates()
static.loc[~(static["is_female"].isin(["Female", "Male"])), "is_female"] = np.nan
static["is_female"] = static["is_female"].replace({"Female": 1, "Male" : 0}).fillna(-1)

```

## 2.5 Extract longitudinal variables, map lab and vitals

The entries from **event_cd** needs be mapped to the correct variable name for the model. Please see the **sepsis_ml_map.csv** for the full list.

```python
sepsis_ml_map = pd.read_csv("sepsis_ml_map.csv")
sepsis_ml_map = sepsis_ml_map[["Category", "event_cd"]]

test_data = test_data.merge(sepsis_ml_map, on = "event_cd", how = "left")
test_data = test_data.pivot_table(index=['csn', "pat_id", "recorded_time"], columns='Category', values='result_val')
test_data = test_data.reset_index()

```
## 2.6 Sparse matrix

Merge static and longitudinal dataframe together to create a sparse matrix

```python

test_data = test_data.merge(static, on = ["csn", "pat_id"], how = "left")

    for c in list(sepsis_ml_map.Category.unique()):
        if not (c in test_data.columns): 
            test_data[c] = np.nan

# we will prioritize line measurements; but if not available, we will use cuff measurements
test_data["SBP"] = test_data["SBP"].fillna(test_data["SBP_Cuff"])
test_data["MAP"] = test_data["MAP"].fillna(test_data["MAP_Cuff"])
test_data["DBP"] = test_data["DBP"].fillna(test_data["DBP_Cuff"])

# here, fibrinogen was part of the extracted data-- but should have been removed from data extraction section
test_data = test_data.drop(["SBP_Cuff", "MAP_Cuff", "DBP_Cuff", "Fibrinogen"], axis = 1)

```
## 3. Feature engineering

From sepsis_ml.py, mainly three types of handcraft features are derived.


### 3.1. Statistical features
The 18-hour moving average, median, maximum, minimum, standard deviation, and difference in standard deviations were calculated for basic vital signs. 

```python
def feature_slide_window(vitals):

    diff = vitals.shift(-1) - vitals
    rolling_mean = vitals.groupby("csn").rolling(6, min_periods = 1).mean().reset_index(drop = True)
    rolling_mean = rolling_mean.rename(columns = {'HR': "HR_mean", 'O2Sat': "O2Sat_mean", 'SBP':"SBP_mean", 
                                                  'MAP': "MAP_mean", 'Resp': "Resp_mean"})

    rolling_median = vitals.groupby("csn").rolling(6, min_periods = 1).median().reset_index(drop = True)
    rolling_median = rolling_median.iloc[:,1:].rename(columns = {'HR': "HR_median", 'O2Sat': "O2Sat_median", 'SBP':"SBP_median", 
                                                  'MAP': "MAP_median", 'Resp': "Resp_median"})

    rolling_min = vitals.groupby("csn").rolling(6, min_periods = 1).min().reset_index(drop = True)
    rolling_min = rolling_min.iloc[:,1:].rename(columns = {'HR': "HR_min", 'O2Sat': "O2Sat_min", 'SBP':"SBP_min", 
                                                  'MAP': "MAP_min", 'Resp': "Resp_min"})

    rolling_max = vitals.groupby("csn").rolling(6, min_periods = 1).max().reset_index(drop = True)
    rolling_max = rolling_max.iloc[:,1:].rename(columns = {'HR': "HR_max", 'O2Sat': "O2Sat_max", 'SBP':"SBP_max", 
                                                  'MAP': "MAP_max", 'Resp': "Resp_max"})

    rolling_std = vitals.groupby("csn").rolling(6, min_periods = 1).std().reset_index(drop = True)
    rolling_std = rolling_std.iloc[:,1:].rename(columns = {'HR': "HR_std", 'O2Sat': "O2Sat_std", 'SBP':"SBP_std", 
                                                  'MAP': "MAP_std", 'Resp': "Resp_std"})

    rolling_diff_std = diff.groupby("csn").rolling(6, min_periods = 1).std().reset_index(drop = True)
    rolling_diff_std = rolling_diff_std.iloc[:,1:].rename(columns = {'HR': "HR_dstd", 'O2Sat': "O2Sat_dstd", 'SBP':"SBP_dstd", 
                                                  'MAP': "MAP_dstd", 'Resp': "Resp_dstd"})

    rolling_vitals = pd.concat([rolling_mean, rolling_median, rolling_min, rolling_max, rolling_std, rolling_diff_std], axis = 1)
    
    
    return rolling_vitals

```



## 3.2. Missing value features and differential features
These features directly proposed by Yang et. al. (doi: 10.1097/CCM.0000000000004550).
- Measurement frequency sequence (_interval_f1): Record the number of variable measurements before the current time
- Measurement time interval sequence (_interval_f2): Record the time interval from the last measurement between the current time. 
- Differential features (_diff): current value – previous last known value

```python

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

```

## 3.3. Emperical scores
- SBP score and RR score (based on qSOFA score)
- Platelets score, Bilirubin score, MAP score, and Creatinine score (based on SOFA score)
- HR score and Temp score (based on NEWS score
- SOFA >=2 indicator

```python
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
```


## 4. Test

The model is presaved in a specified direcotry and is user-defined during the function call. The **predict** function outputs the following:
- SepsisPredictionProbability
- SepsisPredictionLabel
- ForcePlot diagram to *imgcache*
- Ranked top 10 positive impact SHAP features

```python

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
```


## 5. Output Upload

Along with past historical output read, the output of the new predictions are uploaded to the *outputcache* table.




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)