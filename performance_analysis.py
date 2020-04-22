# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:16:13 2019

@author: vmani
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


def get_rdr(od, oc):
    
    # Disc and cup centroid
    dX, dY = utils.get_centroid(od)
    cX, cY = utils.get_centroid(oc)
    
    # Find contour points 
    contoursD, _ = cv2.findContours(od, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoursC, _ = cv2.findContours(oc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Check for multiple outlines 
    #assert len(contoursD) == 1 and  len(contoursC) == 1

    # From list to a 2D list of X, Y coordinates
    contoursD = contoursD[0]
    contoursC = contoursC[0]
    
    x, y, Dw, Dh = cv2.boundingRect(contoursD)
    x, y, Cw, Ch = cv2.boundingRect(contoursC)
    
    # [theta, Cx, Cy, Dx, Dy]
    look_up_table, vCDR = utils.build_look_up_table((cX, cY), contoursC, contoursD)
    
    # params required for computing RDR
    r_w, diag, sweep = utils.get_min_rim_width(look_up_table, contoursC, contoursD, od.copy())
    
    if diag == 0:
        return 0
    
    else:
        # compute RDR
        return round(r_w / diag, 4)


#@jit(parallel=True)
def get_cdr_rdr_plots(gndCupPath, predCupPath, gndDiscPath, predDiscPath, color, met="CDR"):
    
    gndC_ids = natsorted(next(os.walk(gndCupPath))[2])
    predC_ids = natsorted(next(os.walk(predCupPath))[2])

    assert len(gndC_ids) == len(predC_ids)

    gndD_ids = natsorted(next(os.walk(gndDiscPath))[2])
    predD_ids = natsorted(next(os.walk(predDiscPath))[2])

    assert len(gndD_ids) == len(predD_ids)
    
    print("Length: ", len(gndC_ids))
    
    GT = []
    PR = []
    
    for i in range(len(gndC_ids)):

        gndC = cv2.imread(os.path.join(gndCupPath, gndC_ids[i]), 0)
        preC = cv2.imread(os.path.join(predCupPath, predC_ids[i]), 0)
        
        gndD = cv2.imread(os.path.join(gndDiscPath, gndD_ids[i]), 0)
        preD = cv2.imread(os.path.join(predDiscPath, predD_ids[i]), 0)
        
        if np.count_nonzero(gndD) == 0 or np.count_nonzero(preD) == 0:
            continue
        
        if np.count_nonzero(gndC) == 0 or np.count_nonzero(preC) == 0:
            continue
        
        if met == "CDR":
            cdrGT = sqrt(np.count_nonzero(gndC) / np.count_nonzero(gndD))
            cdrPR = sqrt(np.count_nonzero(preC) / np.count_nonzero(preD))
        
            GT.append(cdrGT)
            PR.append(cdrPR)
        
            plt.scatter(cdrGT, cdrPR, c=color, alpha=1, marker='o', facecolors='none', s=20)
        
        else:
            
            rdrGT = get_rdr(gndD, gndC)
            rdrPR = get_rdr(preD, preC)
            
            if rdrGT == 0 or rdrPR == 0:
                continue
            
            GT.append(rdrGT)
            PR.append(rdrPR)
        
            plt.scatter(rdrGT, rdrPR, c=color, alpha=1, marker='o', facecolors='none', s=20)
            
    pcc = round(pearsonr(GT, PR)[0], 4)
    
    return pcc


@jit(parallel=True)
def get_roc_plots(gndCupPath, predCupPath, gndDiscPath, predDiscPath):
    
    gndC_ids = natsorted(next(os.walk(gndCupPath))[2])
    predC_ids = natsorted(next(os.walk(predCupPath))[2])

    assert len(gndC_ids) == len(predC_ids)

    gndD_ids = natsorted(next(os.walk(gndDiscPath))[2])
    predD_ids = natsorted(next(os.walk(predDiscPath))[2])

    assert len(gndD_ids) == len(predD_ids)
    
    print("Length: ", len(gndC_ids))
    
    GTc = []
    GTd = []
    
    PRc = []
    PRd = []
    
    for i in range(len(gndC_ids)):

        gndC = np.asarray(cv2.imread(os.path.join(gndCupPath, gndC_ids[i]), 0)) // 255
        preC = np.asarray(cv2.imread(os.path.join(predCupPath, predC_ids[i]), 0)) / 255.0
        
        gndD = np.asarray(cv2.imread(os.path.join(gndDiscPath, gndD_ids[i]), 0)) // 255
        preD = np.asarray(cv2.imread(os.path.join(predDiscPath, predD_ids[i]), 0)) / 255.0
        
        # To save memory resize and compute score
        gndC = cv2.resize(gndC, (100, 100), cv2.INTER_LINEAR)
        preC = cv2.resize(preC, (100, 100), cv2.INTER_LINEAR)
        
        gndD = cv2.resize(gndD, (100, 100), cv2.INTER_LINEAR)
        preD = cv2.resize(preD, (100, 100), cv2.INTER_LINEAR)
    
        GTc.extend(gndC.flatten())
        GTd.extend(gndD.flatten())
        
        PRc.extend(preC.flatten())
        PRd.extend(preD.flatten())

    fprC, tprC, tC = roc_curve(GTc, PRc)
    fprD, tprD, tD = roc_curve(GTd, PRd)
        
    return fprC, tprC, fprD, tprD
        

def Train_Val_Scores():
    
    s = "OC"
    t = s + "_out_N"
    
    dataset = "REFUGE"
        
    gndPath = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
    predPath = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t
    
    TNt, FPt, FNt, TPt = compute_scores(gndPath, predPath) 
    print("\n---Train Set Scores---")
    get_scores(TNt, FPt, FNt, TPt)
    
    
    gndPath = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
    predPath = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t
    
    TNv, FPv, FNv, TPv = compute_scores(gndPath, predPath) 
    print("\n---Val Set Scores---")
    get_scores(TNv, FPv, FNv, TPv)
    
    print("\n---Overall Set Scores---")
    get_scores(TNt + TNv, FPt + FPv, FNt + FNv, TPt + TPv)
    

def Cross_Scores():
    gndPath = "D:/IISc/Python/MrRCNN-Disc_Cup/Messidor/OD"
    predPath = "D:/IISc/Python/MrRCNN-Disc_Cup/Messidor/OD_out_N"
    
    TN, FP, FN, TP = compute_scores(gndPath, predPath) 
    print(TN, FP, FN, TP)
    print("\n---Overall Set Scores---")
    get_scores(TN, FP, FN, TP)


def plot_datas():
    
    dataset = "RIGA"   

    u = "_out_N"
    
    met = "RDR"
    
    #plt.rcParams['figure.dpi'] = 600
    plt.plot([0, 1], [0, 1], ls='dashed', c='r')  
    
    if u == "_out_N":
        plt.title("MrRCNN - " + dataset)
    else:
        plt.title("MRCNN - " + dataset)
    
    plt.xlabel(met + " Prediction")
    plt.ylabel(met + " GroundTruth")
    
    s = "OC"
    t = s + u

    gndCupPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
    predCupPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

    gndCupPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
    predCupPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t


    s = "OD"
    t = s + u

    gndDiscPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
    predDiscPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

    gndDiscPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
    predDiscPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t

    L = []

    pcc = get_cdr_rdr_plots(gndCupPathT, predCupPathT, gndDiscPathT, predDiscPathT, 'c', met)
    L.append(Line2D([0], [0], markerfacecolor='c', marker='o', lw=0, mec='c', label="Train PCC: " + str(pcc)))

    pcc = get_cdr_rdr_plots(gndCupPathV, predCupPathV, gndDiscPathV, predDiscPathV, 'm', met)
    L.append(Line2D([0], [0], markerfacecolor='m', marker='o', lw=0, mec='m', label="Val PCC: " + str(pcc)))

    plt.legend(handles=L)
    plt.show()
    plt.close()
    

def plot_all_datas():
    
    datasets = ["Drishti", "REFUGE", "RIGA"]   

    u = "_out_N"
    
    met = "CDR"
    
    if met == "CDR":
        c = ['b', 'orange']
    else:
        c = ['c', 'm']
    
    plt.rcParams['figure.dpi'] = 600
    plt.plot([0, 1], [0, 1], ls='dashed', c='r')  
    
    if u == "_out":
        plt.title("MRCNN")
    else:
        plt.title("RED-RCNN")
    
    plt.xlabel(met + " GROUND-TRUTH")
    plt.ylabel(met + " PREDICTION")
    
    L = []
    pccT = []
    pccV = []
    
    for dataset in datasets:
        
        s = "OC"
        t = s + u

        gndCupPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
        predCupPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

        gndCupPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
        predCupPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t


        s = "OD"
        t = s + u

        gndDiscPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
        predDiscPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

        gndDiscPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
        predDiscPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t

    
        pccT.append(get_cdr_rdr_plots(gndCupPathT, predCupPathT, gndDiscPathT, predDiscPathT, c[0], met))
        pccV.append(get_cdr_rdr_plots(gndCupPathV, predCupPathV, gndDiscPathV, predDiscPathV, c[1], met))
    

    L.append(Line2D([0], [0], markerfacecolor=c[0], marker='o', lw=0, mec=c[0], label="Train PCC: " + str(round(sum(pccT) / len(pccT), 4))))
    L.append(Line2D([0], [0], markerfacecolor=c[1], marker='o', lw=0, mec=c[1], label="Val PCC: " + str(round(sum(pccV) / len(pccV), 4))))
    
    plt.legend(handles=L)

    if u == "_out_N":
        plt.savefig("D:/IISc/My Papers/MRCNN - Optic Disc and Cup Segmentation/Plots/pcc-cdr plots/RED_RCNN.png", dpi=600)

    elif u == "_out":
        plt.savefig("D:/IISc/My Papers/MRCNN - Optic Disc and Cup Segmentation/Plots/pcc-cdr plots/MRCNN.png", dpi=600)
    
    
    #plt.show()
    #plt.close()
    
    
def plot_auc():
    
    dataset = "REFUGE"   

    u = "_out_U"
    
    plt.rcParams['figure.dpi'] = 600
    
    if u == "_out_NU":
        plt.title("MrRCNN - " + dataset)
    else:
        plt.title("MRCNN - " + dataset)
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    s = "OC"
    t = s + u

    gndCupPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
    predCupPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

    gndCupPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
    predCupPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t


    s = "OD"
    t = s + u

    gndDiscPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
    predDiscPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

    gndDiscPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
    predDiscPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t
    
    L = []

    fprC, tprC, fprD, tprD = get_roc_plots(gndCupPathT, predCupPathT, gndDiscPathT, predDiscPathT)
    plt.plot(fprC, tprC, c='b', ls='-')
    L.append(Line2D([0], [0], c='b', lw=2, mec='b', ls='-', label="Cup Train AUC: " + str(round(auc(fprC, tprC), 4))))
    
    plt.plot(fprD, tprD, c='c', ls='-')
    L.append(Line2D([0], [0], c='c', lw=2, mec='c', ls='-', label="Disc Train AUC: " + str(round(auc(fprD, tprD), 4))))
    
    
    fprC, tprC, fprD, tprD = get_roc_plots(gndCupPathV, predCupPathV, gndDiscPathV, predDiscPathV)
    plt.plot(fprC, tprC, c='orange', ls='-')
    L.append(Line2D([0], [0], c='orange', lw=2, ls='-', mec='orange', label="Cup Val AUC: " + str(round(auc(fprC, tprC), 4))))
    
    plt.plot(fprD, tprD, c='m', ls='-')
    L.append(Line2D([0], [0], c='m', lw=2, ls='-', mec='m', label="Disc Val AUC: " + str(round(auc(fprD, tprD), 4))))
    
    plt.legend(handles=L, loc="lower right")
    plt.show()
    plt.close()
    
#plot_datas()
#plot_auc()
#Cross_Scores()   
# plot_all_datas()

gndPath = "D:/IISc/Python/Neural Style Transfer/Final_NST/_masks"
predPath = "D:/IISc/Python/Neural Style Transfer/Final_NST/Preds4/new_contentpredictions"
    
TN, FP, FN, TP = compute_scores(gndPath, predPath) 
print("\n---Performance Scores---")
get_scores(TN, FP, FN, TP)