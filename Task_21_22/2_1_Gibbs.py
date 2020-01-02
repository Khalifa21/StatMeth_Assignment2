# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:35:31 2019

@author: clari & Mohamed
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from Task_21_22.motif_generator import generateSequences

def computeNkj(magicWordsMatrix,k,j):
    s = 0
    for i in range(len(magicWordsMatrix)):
        if magicWordsMatrix[i][j] == k:
            s+=1
    return s

def computeMwMatrix(seqList, R, lenMotif):
    magicWords = []
    background = []
    for n,r in enumerate(R):
        mwList = seqList[n][r:r+lenMotif]
        L1 = seqList[n][:r]
        L2 = seqList[n][r+lenMotif:]
        bgList = [*L1, *L2]
        magicWords.append(mwList)
        background.append(bgList)
    return magicWords,background


def computeBk(backgroundMat,k):
    s = 0
    for i in range(len(backgroundMat)):
        for j in range(len(backgroundMat[0])):
            if backgroundMat[i][j] == k:
                s+=1
    return s


def probaDj(alphaListMw, W, magicWordsMatrix, variabList):
    sum_alpha = sum(alphaListMw)
    N = len(magicWordsMatrix)
    prod = 1
    gamma_cst = math.gamma(sum_alpha)/math.gamma((N*W)+ sum_alpha)
    for j in range(W):
        prod_j = 1
        for index,k in enumerate(variabList):
            prod_j *= math.gamma(computeNkj(magicWordsMatrix,k,j) + alphaListMw[index])/math.gamma(alphaListMw[index])
        prod_j = gamma_cst * prod_j

        prod *= prod_j
    return prod


def probaDb(alphaListBg, W, backgroundMat, variabList, M):
    sum_alpha_Bg = sum(alphaListBg)
    N = len(backgroundMat)
    gamma_cst = math.gamma(sum_alpha_Bg)/math.gamma((N*(M-W))+sum_alpha_Bg)
    prod_j = 1
    for index, k in enumerate(variabList):
        prod_j *= math.gamma(computeBk(backgroundMat, k) + alphaListBg[index])/math.gamma(alphaListBg[index])
    prod = gamma_cst * prod_j
    return prod


def Gibbs(alphaListBg, alphaListMw, seqList, nb_iterations, W):
    N = len(seqList)
    M = len(seqList[0])
    variabList = np.unique(seqList)
    #INITIALISATION
    samples = {i:0 for i in range(M-W+1)}
    positions_list = []
    positions_list.append(np.random.randint(0, M-W+1, N))
    for it in range(nb_iterations):
        R = positions_list[-1].copy()
        for n in range(N):
            list_proba = []
            for index in range(M-W+1):
                R[n] = index
                magicWordsMat, backgroundMat = computeMwMatrix(seqList, R, W)
                proba = probaDb(alphaListBg, W, backgroundMat, variabList, M) * probaDj(alphaListMw, W, magicWordsMat, variabList)
                list_proba.append(proba)

            # r_max = np.argmax(list_proba)
            list_proba = np.asarray(list_proba)

            s = float(sum(list_proba))
            list_proba = [p / s for p in list_proba]
            r_max = np.argmax(np.random.multinomial(1, list_proba))
            R[n] = r_max
        positions_list.append(R[:])
    positions_list = [positions_list[j] for j in range(100, nb_iterations, 20)]
    results = {}
    for R in positions_list:
        R_str = ''.join(str(i) for i in R.tolist())
        if results.get(R_str):
            results[R_str] += 1
        else:
            results[R_str] = 1

    plt.plot(positions_list)
    finalR, prob = max(results.items(), key=lambda k: k[1])
    print("Final R {} with count {}".format([int(i) for i in finalR], prob))


def task_21():
    alphaListBg = [1, 1, 1, 1]
    alphaListMw = [1, 7, 10, 2]
    num_seq = 5
    len_seq = 10
    len_motif = 5
    iterations = 1000
    seq_list, start_list = generateSequences(alphaListBg, alphaListMw, num_seq, len_seq, len_motif, saveOpt=False, displayOpt=False)[:2]
    print("real r is ",start_list)
    Gibbs(alphaListBg, alphaListMw, seq_list, iterations, len_motif)
    plt.show()

def task_22():
    with open('data/2_1_alphaBg.txt','r') as alphaBg_file:
        alphaListBg = []
        for element in alphaBg_file:
            alphaListBg.append(float(element[:-1]))

    with open('data/2_1_alphaMw.txt','r') as alphaMw_file:
        alphaListMw = []
        for element in alphaMw_file:
            alphaListMw.append(float(element[:-1]))

    with open('data/2_1_data.txt','r') as data_file:
        seq_list = []
        num_seq = 0
        for line in data_file:
            seq_list.append([ch for ch in (line[:-1].split())])
            num_seq += 1
        seq_list = np.array(seq_list)

    len_motif = 5 # needs to be read from user/file
    iterations = 1000 # needs to be read from user/file
    start_list = [] # needs to be read from user/file
    # seq_list, start_list = generateSequences(alphaListBg, alphaListMw, num_seq, len_seq, len_motif, saveOpt=False, displayOpt=False)[:2]
    print("real r is ",start_list)
    Gibbs(alphaListBg, alphaListMw, seq_list, iterations, len_motif)
    plt.show()

if __name__ == "__main__": task_21()
