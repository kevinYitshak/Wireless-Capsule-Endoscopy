# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:16:13 2019

@author: kevin
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import utils_cdr_rdr as utils

from math import sqrt
from numba import jit
from natsort import natsorted
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc

def get_scores(TN, FP, FN, TP):
    
    sensitivity = TP / (TP + FN)
    print("Sensitivity: ", round(sensitivity, 4))

    specificity = TN / (TN + FP)    
    print("Specificity: ", round(specificity, 4))
    
    error_rate = (FP + FN) / (TP + FN + TN + FP)
    print("Error_rate: ", round(error_rate, 4))
    
    accuracy = 1 - error_rate
    print("Accuracy: ", round(accuracy, 4))
    
    precision = TP / (TP + FP)
    print("Precision: ", round(precision, 4))
    
    dice_coeff = (2 * TP) / (2 * TP + FP + FN)
    print("Dice_Coeff: ", round(dice_coeff, 4))
    
    jaccard = dice_coeff / (2 - dice_coeff)
    print("Jaccard: ", round(jaccard, 4))
    
    f_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    print("F_score: ", round(f_score, 4), "\n")
    
    
@jit(nopython=True)
def confusion_matrix(gnd, pre, TN, FP, FN, TP):
        
    #print(gnd.shape, pre.shape)
    assert gnd.shape == pre.shape
    
    for x in range(gnd.shape[0]):
            
        for y in range(gnd.shape[1]):
                
            if gnd[x][y] == pre[x][y] == 0:
                TN += 1
                    
            elif gnd[x][y] == pre[x][y] == 255:
                TP += 1
                    
            elif gnd[x][y] == 0 and pre[x][y] == 255:
                FP += 1
                
            elif gnd[x][y] == 255 and pre[x][y] == 0:
                FN += 1
            
    
    #print(TN, FP, FN, TP)
    return TN, FP, FN, TP
    

def compute_scores(gndPath, predPath):
    
    TN = 0
    TP = 0
    FN = 0
    FP = 0

    gnd_ids = natsorted(next(os.walk(gndPath))[2])
    
    pred_ids = natsorted(next(os.walk(predPath))[2])

    assert len(gnd_ids) == len(pred_ids)

    print("Length: ", len(gnd_ids))
    
    for i in range(len(gnd_ids)):
        
        print(gnd_ids[i])
        
        gnd = cv2.imread(os.path.join(gndPath, gnd_ids[i]), 0)
        
        pre = cv2.imread(os.path.join(predPath, pred_ids[i]), 0)
        
        # pre[pre < 120] = 0
        # pre[pre >= 120] = 255

        #gnd = cv2.resize(gnd, (pre.shape[1], pre.shape[0]), cv2.INTER_LINEAR)
        
        #cv2.imwrite(os.path.join(gndPath, gnd_ids[i]), gnd)
        
        TN, FP, FN, TP = confusion_matrix(gnd, pre, TN, FP, FN, TP)
        
    return TN, FP, FN, TP

gndPath = "D:/IISc/Python/Neural Style Transfer/Final_NST/_masks"
predPath = "D:/IISc/Python/Neural Style Transfer/Final_NST/Preds4/new_contentpredictions"
    
TN, FP, FN, TP = compute_scores(gndPath, predPath) 
print("\n---Performance Scores---")
get_scores(TN, FP, FN, TP)