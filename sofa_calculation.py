# -*- coding: utf-8 -*-
"""
Elite Data Hacks
@author: Christopher S. Josef, MD
@Emory University
"""

########################################################################################################
####################                  SOFA FUNCTIONS                                 ###################
########################################################################################################

    def SOFA_resp(self,
                  row,
                  pf_pa='pf_pa',
                  pf_sp = 'pf_sp'):
        """Accepts- class instance, one row from "super_table", "pf" cols
        Does- Calculates Respiratory SOFA score
        Returns- Single value of Respiratory SOFA score """
        if row[pf_pa] < 100:
            val = 4
        elif row[pf_pa] < 200:
            val = 3
        elif row[pf_pa] < 300:
            val = 2
        elif row[pf_pa] < 400:
            val = 1
        elif row[pf_pa] >= 400:
            val = 0
        else: 
            val = float("NaN")
        return val
    
    def SOFA_resp_sa(self,
                  row,
                  pf_pa='pf_pa',
                  pf_sp = 'pf_sp'):
        """Accepts- class instance, one row from "super_table", "pf" cols
        Does- Calculates Respiratory SOFA score
        Returns- Single value of Respiratory SOFA score """
        if row[pf_sp] < 67:
            val = 4
        elif row[pf_sp] < 142:
            val = 3
        elif row[pf_sp] < 221:
            val = 2
        elif row[pf_sp] < 302:
            val = 1
        elif row[pf_sp] >= 302:
            val = 0
        else: 
            val = float("NaN")
        return val

    def SOFA_cardio(self,
                    row,
                    dopamine_dose_weight ='dopamine_dose_weight',
                    epinephrine_dose_weight ='epinephrine_dose_weight',
                    norepinephrine_dose_weight  = 'norepinephrine_dose_weight',
                    dobutamine_dose_weight ='dobutamine_dose_weight'):
        """
        Accepts- class instance, one row from "super_table", weight based pressor cols
        Does- Calculates Cardio SOFA score
        Returns- Single value of Cardio SOFA score 
        """
        
        if ((row[dopamine_dose_weight] > 15) |
            (row[epinephrine_dose_weight] > 0.1) | 
            (row[norepinephrine_dose_weight] > 0.1)):
            val = 4
        elif ((row[dopamine_dose_weight] > 5) |
              ((row[epinephrine_dose_weight] > 0.0) & (row[epinephrine_dose_weight] <= 0.1)) | 
              ((row[norepinephrine_dose_weight] > 0.0) & (row[norepinephrine_dose_weight] <= 0.1))):
            val = 3
        elif (((row[dopamine_dose_weight] > 0.0) & (row[dopamine_dose_weight] <= 5))|
              (row[dobutamine_dose_weight] > 0)):
                val = 2
        elif (row['best_map'] < 70):
            val = 1
            
        elif (row['best_map'] >= 70):
            val = 0
        else:
            val = float("NaN")
        return val

    def SOFA_coag(self,
                  row):
        if row['platelets'] >= 150:
            val = 0
        elif (row['platelets'] >= 100) & (row['platelets'] < 150):
            val = 1
        elif (row['platelets'] >= 50) & (row['platelets'] < 100):
            val = 2
        elif (row['platelets'] >= 20) & (row['platelets'] < 50):
            val = 3
        elif (row['platelets'] < 20):
            val = 4
        else:
            val = float("NaN")
        return val

    def SOFA_neuro(self,
                  row):
        if (row['gcs_total_score'] == 15):
            val = 0
        elif (row['gcs_total_score'] >= 13) & (row['gcs_total_score'] <= 14):
            val = 1
        elif (row['gcs_total_score'] >= 10) & (row['gcs_total_score'] <= 12):
            val = 2
        elif (row['gcs_total_score'] >= 6) & (row['gcs_total_score'] <= 9):
            val = 3
        elif (row['gcs_total_score'] < 6):
            val = 4
        else:
            val = float("NaN")
        return val

    def SOFA_hep(self,
                  row):
        if (row['bilirubin_total'] < 1.2):
            val = 0
        elif (row['bilirubin_total'] >= 1.2) & (row['bilirubin_total'] < 2.0):
            val = 1
        elif (row['bilirubin_total'] >= 2.0) & (row['bilirubin_total'] < 6.0):
            val = 2
        elif (row['bilirubin_total'] >= 6.0) & (row['bilirubin_total'] < 12.0):
            val = 3
        elif (row['bilirubin_total'] >= 12.0):
            val = 4
        else:
            val = float("NaN")
        return val

    def SOFA_renal(self,
                  row):
        if (row['creatinine'] < 1.2):
            val = 0
        elif (row['creatinine'] >= 1.2) & (row['creatinine'] < 2.0):
            val = 1
        elif (row['creatinine'] >= 2.0) & (row['creatinine'] < 3.5):
            val = 2
        elif (row['creatinine'] >= 3.5) & (row['creatinine'] < 5.0):
            val = 3
        elif (row['creatinine'] >= 5.0):
            val = 4
        else:
            val = float("NaN")
        return val
    def SOFA_cardio_mod(self,
                    row,
                    dopamine_dose_weight ='dopamine_dose_weight',
                    epinephrine_dose_weight ='epinephrine_dose_weight',
                    norepinephrine_dose_weight  = 'norepinephrine_dose_weight',
                    dobutamine_dose_weight ='dobutamine_dose_weight'):
        """
        Accepts- class instance, one row from "super_table", weight based pressor cols
        Does- Calculates Cardio SOFA score
        Returns- Single value of Cardio SOFA score 
        """
        
        if ((row[epinephrine_dose_weight] > 0.0) & (row[epinephrine_dose_weight] > 0.0)):
            val = 4
        elif ((row[epinephrine_dose_weight] > 0.0) | (row[epinephrine_dose_weight] > 0.0)):
            val = 3
        elif ((row[dopamine_dose_weight] > 0.0) | (row[dobutamine_dose_weight] > 0)):
                val = 2
        elif (row['best_map'] < 70):
            val = 1
        elif (row['best_map'] >= 70):
            val = 0
        else:
            val = float("NaN")
        return val
    
    def calc_all_SOFA(self,
                window = 24):
    
        df = self.super_table
        sofa_df = pd.DataFrame(index = self.super_table.index,
                               columns=[
                               'SOFA_coag',
                               'SOFA_renal',
                               'SOFA_hep',
                               'SOFA_neuro',
                               'SOFA_cardio',
                               'SOFA_cardio_mod',
                               'SOFA_resp',
                               'SOFA_resp_sa'])
        
        sofa_df['SOFA_coag'] = df.apply(self.SOFA_coag, axis=1)
        sofa_df['SOFA_renal'] = df.apply(self.SOFA_renal, axis=1)
        sofa_df['SOFA_hep'] = df.apply(self.SOFA_hep, axis=1)
        sofa_df['SOFA_neuro'] = df.apply(self.SOFA_neuro, axis=1)
        sofa_df['SOFA_cardio'] = df.apply(self.SOFA_cardio, axis=1)
        sofa_df['SOFA_cardio_mod'] = df.apply(self.SOFA_cardio_mod, axis=1)        
        sofa_df['SOFA_resp'] = df.apply(self.SOFA_resp, axis=1)
        sofa_df['SOFA_resp_sa'] = df.apply(self.SOFA_resp_sa, axis=1)
        ######## Normal Calcs                
        # Calculate NOMRAL hourly totals for each row
        sofa_df['hourly_total'] = sofa_df[[
                               'SOFA_coag',
                               'SOFA_renal',
                               'SOFA_hep',
                               'SOFA_neuro',
                               'SOFA_cardio',
                               'SOFA_resp']].sum(axis=1)
        
        # Calculate POST 24hr delta in total SOFA Score
        sofa_df['delta_24h'] = sofa_df['hourly_total'].\
        rolling(window=window, min_periods=24).\
        apply(lambda x: x.max() - x.min() if x.idxmax().value> x.idxmin().value else 0 ).tolist()
 
        # Calculate FIRST 24h delta in total SOFA score
        sofa_df.update(sofa_df.loc[sofa_df.index[0:24],['hourly_total']].\
        rolling(window=window, min_periods=1).max().rename(columns={'hourly_total':'delta_24h'}))

        ######## Modified Calcs                
        # Calculate NOMRAL hourly totals for each row
        sofa_df['hourly_total_mod'] = sofa_df[[
                               'SOFA_coag',
                               'SOFA_renal',
                               'SOFA_hep',
                               'SOFA_neuro',
                               'SOFA_cardio_mod',
                               'SOFA_resp_sa']].sum(axis=1)
        
        # Calculate POST 24hr delta in total SOFA Score
        sofa_df['delta_24h_mod'] = sofa_df['hourly_total_mod'].\
        rolling(window=window, min_periods=24).\
        apply(lambda x: x.max() - x.min() if x.idxmax().value> x.idxmin().value else 0 ).tolist()
 
        # Calculate FIRST 24h delta in total SOFA score
        sofa_df.update(sofa_df.loc[sofa_df.index[0:24],['hourly_total_mod']].\
        rolling(window=window, min_periods=1).max().rename(columns={'hourly_total_mod':'delta_24h_mod'}))                
        


# =============================================================================

#