import sepsis_ml as ml
import numpy as np
import pandas as pd
import argparse



### usage: sepsis_ml_run.py -i <inputfile>


def run(test_data_path, model_path):
    
    dtypes = {"pat_id": int, "csn": int, "HR": float, "O2Sat": float, "Temp": float, 
             "SBP": float, "DBP": float, "Resp": float, "EtCO2": float, 
             "AST": float, "Alkalinephos": float, "BUN": float, "BaseExcess": float,
             "Bilirubin_total": float, "Calcium": float, "Creatinine": float,
             "FiO2": float, "Glucose": float, "HCO3": float, "Hct": float, 
             "Hgb": float, "Lactate": float, "Magnesium": float, "PTT": float,
             "PaCO2": float, "PaO2": float, "Phosphate": float, "Platelets": float,
             "Potassium": float, "SaO2": float, "Sodium": float, "WBC": float,
             "pH": float, "gcs_total_score": float, "age": float, "gender": int}
    
    test_data = pd.read_csv(test_data_path, dtype = dtypes, parse_dates= ["recorded_time", "hospital_admission_date_time"])
    
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
    #test_data.to_csv("processed0.csv", index= False)
    
    test_data = test_data.groupby(["pat_id", "csn"]).apply(lambda v: ml.rolling_overlap(v, 6, variables, 3))
    
    test_data = test_data.drop(["pat_id", "csn"], axis = 1).reset_index(drop = False).rename(columns = {"level_2" : "LOS"})
    
    #test_data.to_csv("processed1.csv", index= False)
    
    test_data = test_data.merge(stat, on = ["pat_id", "csn"], how = "left")
    
    #di = {"Female": 0, "Male": 1}
    #test_data= test_data.replace({"gender": di})
    #test_data["gender"] = test_data["gender"].fillna(-1)
    
    test_data = ml.preprocess(test_data)
    #test_data.to_csv("test_preprocessed.csv", index = False)
    #test_data = pd.read_csv(test_data_path)
    test_set = list(test_data.csn.unique())
    #print(list(test_data.columns))
  
    result = ml.predict(test_set, test_data, model_path, 0.48)
    
    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser("real-time sepsis model run")
    parser.add_argument("input_csv", help = "Directory to the dataset to be run.", type=str)
    parser.add_argument("model_dir", help = "Directory to the pre-trained models.", type=str)
    args = parser.parse_args()

    result = run(args.input_csv, args.model_dir)
    
    result.to_csv("output.csv", index = False)
    print("result outputed")
    
    
    