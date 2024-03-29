import pandas as pd
import numpy as np
import math
from math import isnan
from typing import NamedTuple
pd.set_option('display.max_rows', 500)

__version__ = "1.2.0"

# -*- coding: utf-8 -*-
"""
Copyright 2021 shimst3r
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
sofascore computes the Sepsis-related Organ Failure Assessment (SOFA) score
according to Singer et al.:
    https://doi.org/10.1001%2Fjama.2016.0287
"""

class Condition(NamedTuple):
    mean_arterial_pressure: float
    dobutamine: int 
    dopamine_low: int 
    dopamine_med: int 
    dopamine_high: int 
    epi_low: int 
    epi_high: int
    norep_low: int 
    norep_high: int
    platelets_count: int
    creatinine_level: float
    bilirubin_level: float
    glasgow_coma_scale: int
    partial_pressure_of_oxygen: float
    is_mechanically_ventilated: bool
        

def compute_score_for_cardiovascular_system(
    mean_arterial_pressure, dobutamine, dopamine_low, dopamine_med, dopamine_high, epi_low, epi_high, norep_low, norep_high
) -> int:
    """
    Computes score based on mean arterial pressure or catecholamine therapy.
    """
    if dopamine_high == 1 or epi_high == 1 or norep_high == 1:
        return 4
    if dopamine_med == 1 or epi_low == 1 or norep_low == 1:
        return 3  
    if dobutamine == 1 or dopamine_low == 1:
        return 2
    if mean_arterial_pressure < 70:
        return 1
    return 0


def compute_score_for_coagulation(platelets_count: int) -> int:
    """
    Computes score based on platelets count (unit is number per microliter).
    """
    if platelets_count < 20:
        return 4
    if platelets_count < 50:
        return 3
    if platelets_count < 100:
        return 2
    if platelets_count < 150:
        return 1
    return 0


def compute_score_for_kidneys(creatinine_level: float) -> int:
    """Computes score based on Creatinine level (unit is mg/dl)."""
    if creatinine_level >= 5.0:
        return 4
    if creatinine_level >= 3.5:
        return 3
    if creatinine_level >= 2.0:
        return 2
    if creatinine_level >= 1.2:
        return 1
    return 0


def compute_score_for_liver(bilirubin_level: float) -> int:
    """Computes score based on Bilirubin level (unit is mg/dl)."""
    if bilirubin_level >= 12.0:
        return 4
    if bilirubin_level >= 6.0:
        return 3
    if bilirubin_level >= 2.0:
        return 2
    if bilirubin_level >= 1.2:
        return 1
    return 0


def compute_score_for_nervous_system(glasgow_coma_scale: int) -> int:
    """
    Computes score based on Glasgow Coma Scale, see paper by Teasdale et al.:
        https://doi.org/10.1016/S0140-6736(74)91639-0
    """
    if glasgow_coma_scale < 6:
        return 4
    if glasgow_coma_scale < 10:
        return 3
    if glasgow_coma_scale < 13:
        return 2
    if glasgow_coma_scale < 15:
        return 1
    return 0


def compute_score_for_respiratory_system(
    partial_pressure_of_oxygen: float, is_mechanically_ventilated: bool
) -> int:
    """Computes score based on PaO2 (unit is mmHg)."""
    if partial_pressure_of_oxygen < 100 and is_mechanically_ventilated:
        return 4
    if partial_pressure_of_oxygen < 200 and is_mechanically_ventilated:
        return 3
    if partial_pressure_of_oxygen < 300:
        return 2
    if partial_pressure_of_oxygen < 400:
        return 1
    return 0


def calculate_24_hr_delta(temp, window):
    rolled= temp.copy()
    rolled['SOFA_delta_24h'] = rolled.rolling(window, min_periods = window)['Current_SOFA_Score'].\
    apply(lambda x: x.max() - x.min() if x.idxmax()> x.idxmin() else x.min() - x.max()).tolist()
    rolled = rolled.reset_index(drop = True)
    return rolled

def compute(condition: Condition) -> int:
    if any(isnan(val) for val in condition):
        return np.nan
    
    cvs_score = compute_score_for_cardiovascular_system(
        mean_arterial_pressure=condition.mean_arterial_pressure,
        dobutamine=condition.dobutamine, 
        dopamine_low=condition.dopamine_low,
        dopamine_med=condition.dopamine_med,
        dopamine_high=condition.dopamine_high,
        epi_low=condition.epi_low,
        epi_high=condition.epi_high,
        norep_low=condition.norep_low,
        norep_high=condition.norep_high
    )
    cg_score = compute_score_for_coagulation(platelets_count=condition.platelets_count)
    kdny_score = compute_score_for_kidneys(creatinine_level=condition.creatinine_level)
    livr_score = compute_score_for_liver(bilirubin_level=condition.bilirubin_level)
    ns_score = compute_score_for_nervous_system(
        glasgow_coma_scale=condition.glasgow_coma_scale
    )
    rs_score = compute_score_for_respiratory_system(
        partial_pressure_of_oxygen=condition.partial_pressure_of_oxygen,
        is_mechanically_ventilated=condition.is_mechanically_ventilated,
    )
    #print(cvs_score, cg_score, kdny_score, livr_score, ns_score, rs_score)
    return cvs_score + cg_score + kdny_score + livr_score + ns_score + rs_score