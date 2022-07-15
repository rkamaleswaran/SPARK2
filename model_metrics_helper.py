import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def SIRS(data):
    
    # calculates SIRS score
    
    temp = data[["csn", "pat_id", "rel_time", "HR", "Resp", "Temp", "WBC"]].copy()
    temp["tachycardia"] = (temp["HR"] > 90)
    temp["tachypnea"] = (temp["Resp"] > 20)
    temp["fever/hypothermia"] = (temp["Temp"] > 38) |(temp["Temp"] < 36)
    temp["wbc"] = (temp["WBC"] > 12) | (temp["WBC"] < 4)
    temp.fillna(0)
    temp["SIRS"] = temp[["tachycardia", "tachypnea", "fever/hypothermia", "wbc"]].sum(axis = 1)
    
    return temp[["csn", "pat_id", "rel_time", "SIRS"]]


def EHC_SIRS(data):
    
    # calculates modified SIRS score defined by Emory 
    
    temp = data[["csn", "pat_id", "rel_time", "HR", "Resp", "Temp", "WBC", "gcs_total_score"]].copy()
    temp["tachycardia"] = (temp["HR"] > 110)
    temp["tachypnea"] = (temp["Resp"] > 22)
    temp["fever/hypothermia"] = (temp["Temp"] >= 38) |(temp["Temp"] < 35)
    temp["wbc"] = (temp["WBC"] > 12) | (temp["WBC"] < 4)
    temp["AMS"] = (temp["gcs_total_score"] <= 8)
    temp.fillna(0)
    temp["SIRS"] = temp[["tachycardia", "tachypnea", "fever/hypothermia", "wbc", "AMS"]].sum(axis = 1)
    
    return temp[["csn", "pat_id", "rel_time", "EHC_SIRS"]]


def MEWS(data):
    
    # calculates MEWS score
    
    temp = data[["csn", "pat_id", "rel_time", "HR", "SBP", "Resp", "Temp", "WBC", "gcs_total_score"]].copy()
    
    # Temp
    temp["temp_score"] = 0
    mask = (temp["Temp"] <= 35.1) | (temp["Temp"] >= 38.4)
    temp.loc[mask,"temp_score"] = 2
    
    # SBP
    temp["SBP"] = round(temp["SBP"])
    temp["SBP_score"] = 0
    mask = (temp["SBP"] <= 70)
    temp.loc[mask, "SBP_score"] = 3
    mask = (temp["SBP"] >= 71) & (temp["SBP"] <= 80)
    temp.loc[mask, "SBP_score"] = 2
    mask = (temp["SBP"] >= 81) & (temp["SBP"] <= 100)
    temp.loc[mask, "SBP_score"] = 1
    mask = (temp["SBP"] >= 101) & (temp["SBP"] <= 199)
    temp.loc[mask, "SBP_score"] = 0
    mask = (temp["SBP"] >= 200)
    temp.loc[mask, "SBP_score"] = 2

    
    # HR
    temp["HR"] = round(temp["HR"])
    temp["HR_score"] = 2
    mask = (temp["HR"]>=41) & (temp["HR"]<=50)
    temp.loc[mask, "HR_score"] = 1
    mask = (temp["HR"]>=51) & (temp["HR"]<=100)
    temp.loc[mask, "HR_score"] = 0
    mask = (temp["HR"]>=101) & (temp["HR"]<=110)
    temp.loc[mask, "HR_score"] = 1
    mask = (temp["HR"]>=130)
    temp.loc[mask, "HR_score"] = 3

    
    temp["gcs_total_score"] = round(temp["gcs_total_score"])
    temp["GCS_score"] = 0
    mask = (temp["gcs_total_score"]>=10) & (temp["gcs_total_score"]<=13)
    temp.loc[mask, "GCS_score"] = 1
    mask = (temp["gcs_total_score"]>=4) & (temp["gcs_total_score"]<=9)
    temp.loc[mask, "GCS_score"] = 2
    mask = (temp["gcs_total_score"]<3)
    temp.loc[mask, "GCS_score"] = 3
    
    temp["Resp"] = round(temp["Resp"])
    temp["RR_score"] = 0
    mask = temp["Resp"] <= 9
    temp.loc[mask,"RR_score"] = 2
    mask = (temp["Resp"] >= 15) & (temp["Resp"] <= 20)
    temp.loc[mask,"RR_score"] = 1
    mask = (temp["Resp"] >= 21) & (temp["Resp"] <= 29)
    temp.loc[mask,"RR_score"] = 2
    mask = temp["Resp"] >= 30
    temp.loc[mask,"RR_score"] = 3

    temp["MEWS"] = temp["temp_score"] + temp["SBP_score"] + temp["HR_score"] + temp["GCS_score"] + temp["RR_score"]
    
    return temp[["csn", "pat_id", "rel_time", "MEWS"]]

def EHC_MEWS(data):
    
    # calculates modified MEWS score defined by Emory 
    temp = data[["csn", "pat_id", "rel_time", "HR", "MAP", "Resp", "Temp", "O2Sat"]].copy()
    
    # Temp
    temp["temp_score"] = 0
    mask = (temp["Temp"] < 35) | (temp["Temp"] > 38.5)
    temp.loc[mask,"temp_score"] = 2
    
    # SBP
    temp["MAP"] = round(temp["MAP"])
    temp["MAP_score"] = 0
    mask = (temp["MAP"] <= 70)
    temp.loc[mask, "MAP_score"] = 2
    mask = (temp["MAP"] >= 70) & (temp["MAP"] <= 80)
    temp.loc[mask, "MAP_score"] = 1
    mask = (temp["MAP"] > 130)
    temp.loc[mask, "MAP_score"] = 2

    
    # HR
    temp["HR"] = round(temp["HR"])
    temp["HR_score"] = 0
    mask = (temp["HR"] < 41)
    temp.loc[mask, "HR_score"] = 3
    mask = (temp["HR"]>=41) & (temp["HR"]<=50)
    temp.loc[mask, "HR_score"] = 2
    mask = (temp["HR"]>=101) & (temp["HR"]<=110)
    temp.loc[mask, "HR_score"] = 1
    mask = (temp["HR"]>=111) & (temp["HR"]<=129)
    temp.loc[mask, "HR_score"] = 2
    mask = (temp["HR"]>=130)
    temp.loc[mask, "HR_score"] = 3

    
    temp["Resp"] = round(temp["Resp"])
    temp["RR_score"] = 0
    mask = temp["Resp"] < 9
    temp.loc[mask,"RR_score"] = 2
    mask = (temp["Resp"] >= 15) & (temp["Resp"] <= 20)
    temp.loc[mask,"RR_score"] = 1
    mask = (temp["Resp"] >= 21) & (temp["Resp"] <= 29)
    temp.loc[mask,"RR_score"] = 2
    mask = temp["Resp"] >= 30
    temp.loc[mask,"RR_score"] = 3
    
    
    temp["O2Sat"] = round(temp["O2Sat"])
    temp["O2Sat_score"] = 0
    mask = (temp["O2Sat"] <=91)
    temp.loc[mask, "O2Sat_score"] = 3
    mask = (temp["O2Sat"] >= 92) & (temp["O2Sat"] <= 93)
    temp.loc[mask, "O2Sat_score"] = 2
    mask = (temp["O2Sat"] >= 94) & (temp["O2Sat"] <= 95)
    temp.loc[mask, "O2Sat_score"] = 1
    

    temp["MEWS"] = temp["temp_score"] + temp["MAP_score"] + temp["HR_score"] + temp["O2Sat_score"] + temp["RR_score"]
    
    return temp[["csn", "pat_id", "rel_time", "EHC_MEWS"]]


def roc_pr_curve(result, benchmark):
    testy = result["SepsisLabel"]
    ns_probs = result[benchmark].astype(int)
    #ns_probs = [0 for _ in range(len(testy))]
    # fit a model

    lr_probs = result["PredictedProbability"]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print(benchmark + ": ROC AUC=%.3f" % (ns_auc))
    print('48-hour model: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--',label= benchmark + " ROC curve (area = %0.2f)" % ns_auc)
    pyplot.plot(lr_fpr, lr_tpr, marker='.',label= "ML model ROC curve (area = %0.2f)" % lr_auc)

    pyplot.title("ROC Curve")

    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    
    yhat = result["PredictedSepsisLabel"]
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    
    
    no_precision, no_recall, _ = precision_recall_curve(testy, ns_probs)
    no_f1, no_auc = f1_score(testy, ns_probs), auc(no_recall, no_precision)
    
    # summarize scores
    print('48-hour model: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(testy[testy==1]) / len(testy)
    pyplot.plot(no_recall, no_precision, linestyle='--', label = benchmark + " PR Curve (f1 = %0.2f, area = %0.2f)" % (no_f1, no_auc))
    pyplot.plot(lr_recall, lr_precision, marker='.', label =  "ML model PR Curve (f1 = %0.2f, area = %0.2f)" % (lr_f1, lr_auc))
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    pyplot.title("Precision-Recall Curve")
    # show the plot
    pyplot.show()

    
# https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        #normalized by cases
        group_percentages = ["{0:.2%}".format(value) for value in (cf.transpose()/np.sum(cf, axis = 1)).transpose().flatten()]
        
        #group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[0,0] / sum(cf[:,0])
            recall    = cf[0,0] / sum(cf[0,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap((cf.transpose()/np.sum(cf, axis = 1)).transpose(),
                annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
        
        


            
def metric_null(result, title = None, label = "PredictedSepsisLabel"):

    actual = result["SepsisLabel"]
    predicted = result[label]

    cf_matrix = confusion_matrix(actual,predicted, labels=[1,0])
    print('Confusion matrix : \n',cf_matrix)

    # outcome values order in sklearn
    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : ', "\nTP: ", tp, "\nFN: ", fn, "\nFP: ",fp, "\nTN: ",tn)
    
    

    matrix = classification_report(actual,predicted,labels=[1,0], digits = 4)
    print('Classification report : \n',matrix)

    labels = ["TP", "FN", "FP", "TN"]
    categories = ["sepsis", "no sepsis"]
    
    make_confusion_matrix(cf_matrix, 
                          group_names=labels,
                          categories=categories, title = title)
    
    
    df = result.copy()
    df['fp'] = (df[label] == 1) & (df['SepsisLabel'] == 0)
    df['tn'] = (df[label] == 0) & (df['SepsisLabel'] == 0)
    df['tp'] = (df[label] == 1) & (df['SepsisLabel'] == 1)
    df['fn'] = (df[label] == 0) & (df['SepsisLabel'] == 1)


    alarms_result = df.groupby("csn").sum().reset_index()
    alarms_possible = df.groupby("csn")["rel_time"].count()
    alarms_result = alarms_result[["csn", "SepsisLabel", label, "fp", "tn", "tp", "fn"]].rename(columns = {"label": "Alarms"})
    alarms_result["Possible_Alarms"] = alarms_possible.values
    
    alarms_result["Accuracy"] = (alarms_result["tp"] + alarms_result["tn"])/(alarms_result["tp"] + alarms_result["tn"] + alarms_result["fn"] + alarms_result["fp"])
    alarms_result["PPV"] = alarms_result["tp"] / (alarms_result["tp"] + alarms_result["fp"])
    alarms_result["NPV"] = alarms_result["tn"] / (alarms_result["tn"] + alarms_result["fn"])
    alarms_result["Sensitivity"] = alarms_result["tp"]/ (alarms_result["tp"] + alarms_result["fn"])
    alarms_result["Specificity"] = alarms_result["tn"]/ (alarms_result["tn"] + alarms_result["fp"])
    
    pt_avg_results = alarms_result.fillna(1).mean().round(4)
    
    print("\n average patient level metrics",
          "\n average PPV:", pt_avg_results["PPV"],
          "\n average NPV:", pt_avg_results["NPV"],
          "\n average Specificity:", pt_avg_results["Specificity"],
          "\n average Sensitivity:", pt_avg_results["Sensitivity"],
          "\n"
         )
    

    return(cf_matrix, matrix, alarms_result)


def metric_one(result, title = None, label = "PredictedSepsisLabel"):
    
    actual = result["SepsisLabel"]
    predicted = result[label]
    
    cf_matrix_org = confusion_matrix(actual,predicted, labels=[1,0])
    
    df = result.copy()
    df["shifted_label"] = df["SepsisLabel"]
    df['SepsisLabel_temp'] = df.groupby(["csn", "pat_id"])['shifted_label'].shift(-2).fillna(method = "ffill")
    df["SepsisLabel_temp"] = df["SepsisLabel_temp"].fillna(df["shifted_label"])

    mask = (df["SepsisLabel_temp"] == 1) & (df[label] == 1)
    df.loc[mask,"SepsisLabel"] = 1
    
    cf_matrix, matrix, alarms_result = metric_null(df, title, label = label)
    
    diff = cf_matrix - cf_matrix_org
    print("changes from null: \n", diff)
    
    
    return(cf_matrix, matrix, alarms_result)
    

    
def metric_two(result, mode = 1, title = None, label = "PredictedSepsisLabel"):
    
    df = result.copy()
    
    if mode == 1:
        
        df["shifted_label"] = df["SepsisLabel"]
        df['SepsisLabel_temp'] = df.groupby(["csn", "pat_id"])['shifted_label'].shift(-2).fillna(method = "ffill")
        df["SepsisLabel_temp"] = df["SepsisLabel_temp"].fillna(df["shifted_label"])

        mask = (df["SepsisLabel_temp"] == 1) & (df[label] == 1)
        df.loc[mask,"SepsisLabel"] = 1
    
    df["mask"] = (df["SepsisLabel"] == df[label])
    df['mask_shift'] = df.groupby(["csn", "pat_id"])['mask'].shift(1)
    df["stoppers"] = (df["mask"] & df["mask_shift"]) & (df["SepsisLabel"] == 1) & (df[label] == 1)
    v1 = df.groupby(["csn", "pat_id"]).stoppers.idxmax().values
    
    df.loc[v1, "stoppers2"] = (df.loc[v1, "stoppers"] == True)
    df["stoppers2"] = df["stoppers2"].replace(False, np.nan)
    df["stoppers3"] = df.groupby(["csn", "pat_id"])["stoppers2"].fillna(method = "ffill")
    df.loc[df["stoppers2"] == True, "stoppers3"] = False
    df["stoppers3"] = df["stoppers3"].fillna(False)
    
    
    new_df = df.loc[df["stoppers3"] == False,:].copy()
    
    
    cf_matrix, matrix, alarms_result = metric_null(new_df, title, label = label)
    
    
    return(cf_matrix, matrix, alarms_result)




# patient level

def metrics_by_bed_loc(result, mode = 1, label = "PredictedSepsisLabel"):
    grouped = result.groupby(by = "BED")
    
    for name, group in grouped:
        print(name)
        if mode == 0:
            cf_matrix, matrix, alarms_result = metric_null(group, title = name, label = label)
        elif mode == 1:
            cf_matrix, matrix, alarms_result = metric_one(group, title = name, label = label)
        elif mode == 2:
            cf_matrix, matrix, alarms_result = metric_two(group, title = name, label = label)
    return cf_matrix, matrix, alarms_result


def plot_median_first_alert_time(final_df_w_poa, final_df_no_poa_1, final_df_no_poa_12, data_sources, plot_thresholds):
    """
    Code for plotting medan first alert time
    
    """
    
    plot_thresholds = ['ML_' + str(threshold)if threshold not in ['SIRS', 'MEWS'] 
                           else threshold for threshold in plot_thresholds]
    
    select_cols = ['Threshold', 'Median_First_Alert_Time_From_TimeZero', 'ppv']
    
    final_df_w_poa = final_df_w_poa[final_df_w_poa.Threshold.isin(plot_thresholds)][select_cols]
    final_df_w_poa['Data Source'] = data_sources[0]
    
    final_df_no_poa_1 = final_df_no_poa_1[final_df_no_poa_1.Threshold.isin(plot_thresholds)][select_cols]
    final_df_no_poa_1['Data Source'] = data_sources[1]
    
    final_df_no_poa_12 = final_df_no_poa_12[final_df_no_poa_12.Threshold.isin(plot_thresholds)][select_cols]
    final_df_no_poa_12['Data Source'] = data_sources[2]
    
    plot_df = pd.concat([final_df_w_poa, final_df_no_poa_1, final_df_no_poa_12], axis=0)
    plot_df.rename(columns={'Threshold':'SepsisAlertMethod'}, inplace=True)
    plot_df['ppv'] = plot_df['ppv']*100
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import math
    sns.set_style("white")
    sns.color_palette("pastel")
    fig = plt.figure(figsize=(15, 10)) 
    ax = sns.barplot(x = 'Median_First_Alert_Time_From_TimeZero', y = 'SepsisAlertMethod', hue='Data Source',
                data = plot_df, palette = "Set2", edgecolor='black')
    plt.axvline(0, color='red', linewidth = 2, linestyle='solid', label='Time Zero')
    plt.xlabel('Median First Sepsis Alert to/From Time Zero (Hours)', size=18), plt.ylabel('Sepsis Alert Method', size=18)
    plt.xticks(np.arange(math.ceil(plot_df['Median_First_Alert_Time_From_TimeZero'].min()) 
                         + -1, plot_df['Median_First_Alert_Time_From_TimeZero'].max() + 1, 2))
    plt.legend(title="Inclusion Criteria", fontsize=14, title_fontsize=14)
    plt.xticks(fontsize=14), plt.yticks(fontsize=14)
    plt.title('Median First Sepsis Alert Time to/from Time Zero', fontweight="bold", size=20)
    
    for ppv, p in zip(plot_df['ppv'].tolist(), ax.patches):
        if p.get_width() < 0:
            add_x = -0.6
        else:
            add_x = 0.6
        ax.annotate(str(format(ppv, '.1f')) + '%\nPPV', 
                    (p.get_width() + add_x, p.get_y() + (p.get_height()/ 2)), ha = 'center', 
                    va = 'center', fontsize=8, fontweight='bold')
    plt.margins(x=0.05, y=0.01)
    plt.tight_layout()
    plt.grid()
    plt.show() 
            
