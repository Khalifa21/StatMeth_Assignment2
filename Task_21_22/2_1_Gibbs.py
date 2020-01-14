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
    prod = 0
    gamma_cst = math.lgamma(sum_alpha)-math.lgamma((N*W)+sum_alpha)
    for j in range(W):
        prod_j = 0
        for index,k in enumerate(variabList):
            prod_j += math.lgamma(computeNkj(magicWordsMatrix,k,j) + alphaListMw[index])-math.lgamma(alphaListMw[index])
        prod_j = gamma_cst + prod_j
        prod += prod_j

    return prod


def probaDb(alphaListBg, W, backgroundMat, variabList, M):
    sum_alpha_Bg = sum(alphaListBg)
    N = len(backgroundMat)
    gamma_cst = math.lgamma(sum_alpha_Bg)-math.lgamma((N*(M-W))+sum_alpha_Bg)
    prod_j = 0
    for index, k in enumerate(variabList):
        prod_j += math.lgamma(computeBk(backgroundMat, k) + alphaListBg[index])-math.lgamma(alphaListBg[index])
    prod = gamma_cst + prod_j
    return prod


def Gibbs(alphaListBg, alphaListMw, seqList, nb_iterations, W, burn_in, step):
    N = len(seqList)
    M = len(seqList[0])
    variabList = np.unique(seqList)
    #INITIALISATION
    samples = {i:0 for i in range(M-W+1)}
    positions_list = []
    positions_list.append(np.random.randint(0, M-W+1,size= N))
    for it in range(nb_iterations):
        R = positions_list[-1].copy()
        for n in range(N):
            list_proba = []
            for index in range(M-W+1):
                R[n] = index
                magicWordsMat, backgroundMat = computeMwMatrix(seqList, R, W)
                proba = probaDb(alphaListBg, W, backgroundMat, variabList, M) + probaDj(alphaListMw, W, magicWordsMat, variabList)
                list_proba.append(proba)

            # r_max = np.argmax(list_proba)
            list_proba = np.asarray(list_proba)
            list_proba = np.exp(list_proba - np.max(list_proba))
            list_proba = list_proba / np.sum(list_proba)
            # s = float(sum(list_proba))
            # list_proba = [p / s for p in list_proba]
            r_max = np.argmax(np.random.multinomial(1, list_proba))
            # r_max = np.argmax(list_proba)
            R[n] = r_max
        positions_list.append(R[:])
    positions_list = [positions_list[j] for j in range(burn_in, nb_iterations, step)]
    results = {}
    for R in positions_list:
        R_str = ','.join(str(i) for i in R.tolist())
        if results.get(R_str):
            results[R_str] += 1
        else:
            results[R_str] = 1

    # plt.plot(positions_list)
    positions_list = np.array(positions_list)
    finalR, prob = max(results.items(), key=lambda k: k[1])
    print("Final R {} with count {}".format([int(i) for i in finalR.split(',')], prob))
    return positions_list


def mean_square_error(y_true,y_pred_list):
    output = []
    for i in range(len(y_pred_list)):
        output.append(((y_pred_list[:i]-y_true)**2).mean())
    return output


def estimated_potential_scale_reduction(positions_list):
    """
    reference : https://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf
    """
    m = positions_list.shape[0]
    n = positions_list.shape[1]
    seq_mean = np.mean(positions_list, axis=1)

    trials_mean = np.mean(seq_mean, axis=0)

    B = n / (m - 1.) * np.sum((seq_mean - trials_mean) ** 2, axis=0)
    sj_square = 1. / (n - 1) * np.sum(np.array([(positions_list[trial, :, :] - seq_mean[trial, :]) ** 2 for trial in range(m)]), axis=1)
    W = 1. / m * np.sum(sj_square)
    Var_theta = (1 - 1./n) * W + 1. / n * B

    R = np.sqrt(Var_theta / W)

    return R

def task_21():
    alphaListBg = [1, 1, 1, 1]
    alphaListMw = [.8, .8, .8, .8]
    num_seq = 10
    len_seq = 10
    len_motif = 5
    iterations = 1000
    burn_in = 50
    step = 20
    number_trials = 10

    seq_list, start_list = generateSequences(alphaListBg, alphaListMw, num_seq, len_seq, len_motif, saveOpt=False, displayOpt=False)[:2]
    print("Real r is ", start_list)
    positions_list = []
    for i in range(number_trials):
        positions_list.append(Gibbs(alphaListBg, alphaListMw, seq_list, iterations, len_motif, burn_in, step))
    positions_list = np.array(positions_list)


    #### Predicted Positions #####
    for m in range(num_seq):
        fig = plt.figure(figsize=(16, 8), dpi=360)
        fig.suptitle('Prediction for position {}'.format(m), fontsize=20)
        plt.xlabel('iterations', fontsize=18)
        plt.ylabel('predicted position', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        for trial in range(number_trials):
            plt.plot(positions_list[trial, :, m], label="trial {}".format(trial))
        plt.legend(loc="upper right")
        plt.savefig('Figures/predicted_pos{}.png'.format(m))

    ### Mean Square Error #####
    for m in range(num_seq):
        fig = plt.figure(figsize=(16, 8), dpi=360)
        fig.suptitle('MSE for position {}'.format(m), fontsize=20)
        plt.xlabel('iterations', fontsize=18)
        plt.ylabel('Mse', fontsize=16)
        for trial in range(number_trials):
            plt.plot(mean_square_error(start_list[m], positions_list[trial, :, m]), label="trial {}".format(trial))
        plt.legend(loc="upper right")
        plt.savefig('Figures/mse_pos{}.png'.format(m))

    #### Estimated Potential Scale Reduction #####
    conv = []
    for it in range(2, iterations):
        conv.append(estimated_potential_scale_reduction(positions_list[:, 0:it, :]))
    conv = np.array(conv)
    plt.figure(figsize=(16, 8), dpi=360)
    for m in range(num_seq):
        plt.plot(conv[:, m], label='$position = {}$'.format(m))
    plt.title('Estimated potential scale reduction over sample number')
    plt.legend(loc=1)
    plt.savefig('Figures/estimated_potential_scale_reduction.png')


    plt.show()


def task_22():
    with open('data/2_1_alphaBg.txt','r') as alphaBg_file:
        print("reading alpha Backgoround")
        alphaListBg = []
        for element in alphaBg_file:
            alphaListBg.append(float(element[:-1]))

    with open('data/2_1_alphaMw.txt','r') as alphaMw_file:
        print("reading alpha Magic")
        alphaListMw = []
        for element in alphaMw_file:
            alphaListMw.append(float(element[:-1]))

    with open('data/2_1_data.txt','r') as data_file:
        print("reading data")
        seq_list = []
        num_seq = 0
        for line in data_file:
            seq_list.append([ch for ch in (line[:-1].split())])
            num_seq += 1
            if num_seq ==24:
                break
        seq_list = np.array(seq_list)

    len_motif = 5 # needs to be read from user/file
    iterations = 500
    burn_in = 50
    step = 20
    number_trials = 10
    # start_list = [] # needs to be read from user/file
    # seq_list, start_list = generateSequences(alphaListBg, alphaListMw, num_seq, len_seq, len_motif, saveOpt=False, displayOpt=False)[:2]
    # print("real r is ",start_list)
    positions_list = []
    for i in range(number_trials):
        print("gibbs number",i)
        positions_list.append(Gibbs(alphaListBg, alphaListMw, seq_list, iterations, len_motif, burn_in, step))

    positions_list = np.array(positions_list)


    #### Predicted Positions #####
    for m in range(num_seq):
        fig = plt.figure(figsize=(16, 8), dpi=360)
        fig.suptitle('Prediction for position {}'.format(m), fontsize=20)
        plt.xlabel('iterations', fontsize=18)
        plt.ylabel('predicted position', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        for trial in range(number_trials):
            plt.plot(positions_list[trial, :, m], label="trial {}".format(trial))
        plt.legend(loc="upper right")
        plt.savefig('Figures/q2predicted_pos{}.png'.format(m))

    #### Estimated Potential Scale Reduction #####
    conv = []
    for it in range(2, iterations):
        conv.append(estimated_potential_scale_reduction(positions_list[:, 0:it, :]))
    conv = np.array(conv)
    plt.figure(figsize=(16, 8), dpi=360)
    for m in range(num_seq):
        plt.plot(conv[:, m], label='$position = {}$'.format(m))
    plt.title('Estimated potential scale reduction over sample number')
    plt.legend(loc=1)
    plt.savefig('Figures/q2estimated_potential_scale_reduction.png')

    plt.show()


if __name__ == "__main__": task_22()
