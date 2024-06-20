import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def calc_model_metrics(df, target_col, pred_col, threshold=0.5):
    """
    Calculate and spit out metrics for this project
    """
    y_true = df[target_col].fillna(0)
    y_pred = df[pred_col]
    preds = pd.Series(np.where(y_pred>threshold, 1.0, 0.0))
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, y_pred)

    together = pd.DataFrame({"y_true":y_true, "y_pred":y_pred})
    srtd = together.sort_values(by="y_pred", axis=0).reset_index()
    top5_true = srtd.loc[0:49,:]['y_true']
    #top5_pred = srtd.loc[0:49,:]['y_pred']
    #preds = pd.Series(np.where(top5_pred>threshold, 1.0, 0.0))
    preds = np.ones(50)
    top50_acc = accuracy_score(top5_true, preds)

    srtd = together.sort_values(by="y_pred", axis=0, ascending=True).reset_index()
    recs = int(len(together)/2)
    bot_true = srtd.loc[0:recs-1,:]['y_true']
    preds = np.zeros(recs)
    bot_acc = accuracy_score(bot_true, preds)

    recs = int(2*len(together)/10)
    bot_true = srtd.loc[0:recs-1,:]['y_true']
    preds = np.zeros(recs)
    bot20_acc = accuracy_score(bot_true, preds)

    return {
        "ACC": acc,
        "Top50ACC": top50_acc,
        "Bot50%ACC": bot_acc,
        "Bot20%ACC": bot20_acc,
        "AUC": auc
    }

