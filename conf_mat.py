from sklearn.metrics import confusion_matrix

import torch 
   
def metrics(pred, target, threshold=0.5):

    pred = (pred > threshold).float()
    pred = pred.detach().cpu().numpy() 
    target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    TN, FP, FN, TP = confusion_matrix(target, pred).ravel()

    SE = TP / (TP + FN)

    SPE = TN / (TN + FP)

    error_rate = (FP + FN) / (TP + FN + TN + FP)

    ACC = 1 - error_rate

    dice = (2 * TP) / (2 * TP + FP + FN)

    return SE, SPE, ACC, dice
