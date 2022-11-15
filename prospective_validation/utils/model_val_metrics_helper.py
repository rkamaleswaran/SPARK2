import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef, f1_score, accuracy_score, roc_auc_score, make_scorer, average_precision_score, recall_score, confusion_matrix, precision_recall_curve
import seaborn as sns
import numpy as np

def MEWS(data):
    
    # calculates MEWS score
    
    temp = data[["ID", "rel_time", "HR", "SBP", "Resp", "Temp", "WBC", "gcs_total_score"]].copy()
    
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
    
    return temp[["ID", "rel_time", "MEWS"]]

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
            stats_text = "\n\nAccuracy={:0.2f}\nPrecision={:0.2f}\nRecall={:0.2f}\nF1 Score={:0.2f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.2f}".format(accuracy)
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
    plt.show()
                    
def metric_null(threshold, result, title = None, label = "Sepsis_Case"):

    actual = result[label]
    predicted = result[f"SepsisLabel_{threshold}"]

    cf_matrix = confusion_matrix(actual,predicted, labels=[1,0])
    #print('Confusion matrix : \n',cf_matrix)

    # outcome values order in sklearn
    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    #print('Outcome values : ', "\nTP: ", tp, "\nFN: ", fn, "\nFP: ",fp, "\nTN: ",tn)
    
    

    matrix = classification_report(actual,predicted,labels=[1,0], digits = 4)
    #print('Classification report : \n',matrix)

    labels = ["TP", "FN", "FP", "TN"]
    categories = ["sepsis", "no sepsis"]
    
    make_confusion_matrix(cf_matrix, 
                          group_names=labels, title=title)

def select_threshold(threshold, pred_df):
    sepsis_df = pred_df[pred_df[f'SepsisLabel_{threshold}'] == 1]
    sepsis_df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    
    non_sepsis_df = pred_df[~pred_df['ID'].isin(sepsis_df['ID'].tolist())]
    non_sepsis_df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    
    full_df = pd.concat([sepsis_df, non_sepsis_df])
        
    return full_df


def plot_intervention(plot_df, threshold):
    plot_df = plot_df[plot_df['Threshold'] == f'SepsisLabel_{threshold}']
    plot_df.rename(columns={'Label':'Sepsis_Threshold'}, inplace=True)
    sepsis_threshold = {1:f'>{threshold}', 0:f'<={threshold}'}
    plot_df['Sepsis_Threshold'] = plot_df['Sepsis_Threshold'].apply(lambda x: sepsis_threshold[x])
    plot_df.sort_values(by=['Sepsis_Counts', 'Sepsis_Threshold'], ascending=False, inplace=True)
    plot_df.index = plot_df['Intervention'] + plot_df['Sepsis_Threshold']
    plot_df = plot_df.reindex([f'Intervention Grouper>{threshold}', f'Intervention Grouper<={threshold}', 
                 f'Abx Administration>{threshold}',  f'Abx Administration<={threshold}',
                 f'Lactate Lab Order>{threshold}', f'Lactate Lab Order<={threshold}', 
                 f'IMV>{threshold}', f'IMV<={threshold}'])
    
    sns.set_style("white")
    sns.color_palette("pastel")
    fig = plt.figure(figsize=(10, 7)) 
    ax = sns.barplot(x = 'Intervention', y = 'Sepsis_Counts', hue='Sepsis_Threshold',
                data = plot_df, palette = "Set2", edgecolor='black')
    plt.xlabel('Intervention', size=10), plt.ylabel('Number of Cases', size=10)
    plt.yticks(np.arange(0, plot_df['Sepsis_Counts'].max(), 200))
    plt.legend(title=f" ML Sepsis Score \n @{threshold} threshold", fontsize=14, title_fontsize=14)
    plt.xticks(fontsize=10), plt.yticks(fontsize=10)
    plt.title('Distribution of Sepsis-Related Intervention', fontweight="bold", size=14)
    
    for container in ax.containers:
        ax.bar_label(container, fontweight="bold", size=10)
    plt.margins(x=0.05, y=0.03)
    plt.tight_layout()
    plt.grid()
    plt.show() 
    
def plot_intervention_counts(plot_df):
    #plot_df.index = plot_df['Intervention'] + plot_df['Sepsis_Grouper']
    #high_risk_label = plot_df[plot_df['Sepsis_Grouper'].str.startswith('High')].Sepsis_Grouper.unique()[0]
    #med_risk_label = plot_df[plot_df['Sepsis_Grouper'].str.startswith('Medium')].Sepsis_Grouper.unique()[0]
    #plot_df = plot_df.reindex([intervention + risk for intervention in interventions 
    #                           for risk in [high_risk_label, med_risk_label]])
    plot_df.sort_values(by=['Sepsis_Grouper'], inplace=True)
    sns.set_style("white")
    fig = plt.figure(figsize=(15, 10)) 
    ax = sns.barplot(x = 'Counts', y = 'Intervention', hue='Sepsis_Grouper',
                data = plot_df, palette = "OrRd_r", edgecolor='black')
    plt.xlabel('Number of Cases', size=12),plt.ylabel('', size=12)
    plt.xticks(np.arange(0, plot_df['Counts'].max() + 100, 100))
    plt.legend(title=f" ML Sepsis Risk", fontsize=14, title_fontsize=14)
    plt.xticks(fontsize=12), plt.yticks(fontsize=12)
    plt.title('Distribution of Sepsis-Related Intervention', fontweight="bold", size=18)
    
    for group, p in zip(plot_df['Sepsis_Grouper'].tolist(), ax.patches):
        add_x = 40
        denom = int(group.split('N=')[1])
        val = str(format(p.get_width(), '.0f')) + '\n (' + str(format((p.get_width()/denom)*100 , '.0f')) + '%' + ')'
        ax.annotate((val), 
                    (p.get_width() + add_x, p.get_y() + (p.get_height()/ 2)), ha = 'center', 
                    va = 'center', fontsize=12, fontweight='bold')
    
    plt.margins(x=0.08, y=0.05)
    plt.tight_layout()
    plt.grid()
    plt.show() 