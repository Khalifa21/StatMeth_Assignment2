# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:35:31 2019

@author: clari & Mohamed
"""
import numpy as np
import math
import matplotlib.pyplot as plt

def generateSequences(alphabg_list, alphamw_list, num_seq, len_seq, len_motif, saveOpt=False, displayOpt=False):
    # Generate thetas for Background and Motif with corresponding Dirichlet priors
    thetaBg = np.random.dirichlet(alphabg_list)
    thetaMw = np.random.dirichlet(alphamw_list, size=len_motif) # Generates Theta_j for each motif position

    seqList = np.zeros((num_seq, len_seq))
    startList = np.zeros(num_seq)

    for s in range(num_seq):
        # Get the starting point of motif
        r = np.random.randint(len_seq-len_motif+1)
        startList[s] = r

        for pos in range(len_seq):
            # Sample from Background
            if pos < r or pos >= r+len_motif:
                value = np.where(np.random.multinomial(1,thetaBg)==1)[0][0]
            # Sample from Motif
            else:
                j = pos - r # index of motif letter
                value = np.where(np.random.multinomial(1,thetaMw[j])==1)[0][0]

            seqList[s,pos] = value

    seqList = seqList.astype(int)
    startList = startList.astype(int)

    # Store the motifs in the sequences into a multidimensional array for debugging.
    motifList = np.zeros((num_seq,len_motif))
    for i in range(num_seq):
        r = startList[i]
        motifList[i] = seqList[i,r:r+len_motif]
    motifList = motifList.astype(int)

    if displayOpt:
        print("Background Parameters")
        print("Alpha")
        print(alphabg_list)
        print("Theta")
        print(thetaBg)

        print("\nSequence List")
        print(seqList)
        print("\nStarting Positions of Motifs")
        print(startList)

        print("\nMotifs")
        print(motifList)

        print("\nMotif Parameters")
        print("Alpha")
        print(alphamw_list)
        print("Theta")
        print(thetaMw)

    if saveOpt:
        filename = "data/alphaBg.txt"
        np.savetxt(filename, alphabg_list, fmt='%.5f')

        filename = "data/alphaMw.txt"
        np.savetxt(filename, alphamw_list, fmt='%.5f')

        filename = "data/thetaBg.txt"
        np.savetxt(filename, thetaBg, fmt='%.5f')

        filename = "data/thetaMw.txt"
        np.savetxt(filename, thetaMw, fmt='%.5f')

        filename = "data/sequenceList.txt"
        np.savetxt(filename, seqList, fmt='%d')

        filename = "data/startList.txt"
        np.savetxt(filename, startList, fmt='%d')

        filename = "data/motifsInSequenceList.txt"
        np.savetxt(filename, motifList, fmt='%d')

    return  seqList, startList, motifList, thetaBg, thetaMw

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
    plt.show()

    # finalR = []
    # for index in results:
    #     finalR.append(max(results[index].items(), key=lambda k: k[1])[0])
    finalR, prob = max(results.items(), key=lambda k: k[1])
    print("Final R {} with count {}".format([int(i) for i in finalR], prob))
    # print("Final R {}".format([int(i) for i in finalR]))


def main():
    alphaListBg = [1, 1, 1, 1]
    alphaListMw = [1, 7, 10, 2]
    num_seq = 5
    len_seq = 10
    len_motif = 5
    iterations = 1000
    seq_list, start_list = generateSequences(alphaListBg, alphaListMw, num_seq, len_seq, len_motif, saveOpt=False, displayOpt=False)[:2]
    Gibbs(alphaListBg, alphaListMw, seq_list, iterations, len_motif)
    print("answer", start_list)

if __name__ == "__main__": main()
