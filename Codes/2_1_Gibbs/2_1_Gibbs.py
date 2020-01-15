# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:35:31 2019

@author: clari & Mohamed
"""
import math
import matplotlib.pyplot as plt
import numpy as np
# from Task_21_22.motif_generator import generateSequences
from tqdm import tqdm


def generateSequences(alphaListBg, alphaListMw, numSeq, lenSeq, lenMotif, saveOpt=True, displayOpt=False):
    # Generate thetas for Background and Motif with corresponding Dirichlet priors
    thetaBg = np.random.dirichlet(alphaListBg)
    thetaMw = np.random.dirichlet(alphaListMw, size=lenMotif)  # Generates Theta_j for each motif position

    seqList = np.zeros((numSeq, lenSeq))
    startList = np.zeros(numSeq)

    for s in range(numSeq):
        # Get the starting point of motif
        r = np.random.randint(lenSeq - lenMotif + 1)
        startList[s] = r

        for pos in range(lenSeq):
            # Sample from Background
            if pos < r or pos >= r + lenMotif:
                value = np.where(np.random.multinomial(1, thetaBg) == 1)[0][0]
            # Sample from Motif
            else:
                j = pos - r  # index of motif letter
                value = np.where(np.random.multinomial(1, thetaMw[j]) == 1)[0][0]

            seqList[s, pos] = value

    seqList = seqList.astype(int)
    startList = startList.astype(int)

    # Store the motifs in the sequences into a multidimensional array for debugging.
    motifList = np.zeros((numSeq, lenMotif))
    for i in range(numSeq):
        r = startList[i]
        motifList[i] = seqList[i, r:r + lenMotif]
    motifList = motifList.astype(int)

    if displayOpt:
        print("Background Parameters")
        print("Alpha")
        print(alphaListBg)
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
        print(alphaListMw)
        print("Theta")
        print(thetaMw)

    if saveOpt:
        filename = "data/alphaBg.txt"
        np.savetxt(filename, alphaListBg, fmt='%.5f')

        filename = "data/alphaMw.txt"
        np.savetxt(filename, alphaListMw, fmt='%.5f')

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

    return seqList, startList, motifList, thetaBg, thetaMw


def computeNkj(magicWordsMatrix, k, j):
    s = 0
    for i in range(len(magicWordsMatrix)):
        if magicWordsMatrix[i][j] == k:
            s += 1
    return s


def computeMwMatrix(seqList, R, lenMotif):
    magicWords = []
    background = []
    for n, r in enumerate(R):
        mwList = seqList[n][r:r + lenMotif]
        L1 = seqList[n][:r]
        L2 = seqList[n][r + lenMotif:]
        bgList = [*L1, *L2]
        magicWords.append(mwList)
        background.append(bgList)
    return magicWords, background


def computeBk(backgroundMat, k):
    s = 0
    for i in range(len(backgroundMat)):
        for j in range(len(backgroundMat[0])):
            if backgroundMat[i][j] == k:
                s += 1
    return s


def probaDj(alphaListMw, W, mw_count, variabList, N):
    sum_alpha = sum(alphaListMw)
    prod = 0
    gamma_cst = math.lgamma(sum_alpha) - math.lgamma((N * W) + sum_alpha)
    for j in range(W):
        prod_j = 0
        for index, k in enumerate(variabList):
            prod_j += math.lgamma(mw_count[j][k] + alphaListMw[index]) - math.lgamma(alphaListMw[index])
        prod_j = gamma_cst + prod_j
        prod += prod_j

    return prod


def probaDb(alphaListBg, W, bg_count, variabList, N, M, mw_count, seqList, R, row_number, col_number):
    sum_alpha_Bg = sum(alphaListBg)
    gamma_cst = math.lgamma(sum_alpha_Bg) - math.lgamma((N * (M - W)) + sum_alpha_Bg)
    prod_j = 0
    for index, k in enumerate(variabList):
        try:
            prod_j += math.lgamma(bg_count[k] + alphaListBg[index]) - math.lgamma(alphaListBg[index])
        except:
            # print ("seqlist", seqList)
            # print("row_number",row_number)
            # print("col_number",col_number)
            # print("bgcount", bg_count)
            # print("mwcount", mw_count)
            # print("R",R)
            prod_j += math.lgamma(bg_count[k] + alphaListBg[index]) - math.lgamma(alphaListBg[index])
    prod = gamma_cst + prod_j
    return prod


def init_count(seqList, R, W, variabList):
    mw_count = {}
    bg_count = {}
    ### init ###
    for var in variabList:
        bg_count[var] = 0

    for i in range(W):
        mw_count[i] = {}
        for var in variabList:
            mw_count[i][var] = 0

    ### counting ###
    for row in range(len(seqList)):
        for col in range(len(seqList[0])):
            if col >= R[row] and col < R[row] + W:
                mw_count[col - R[row]][seqList[row][col]] += 1
            else:
                bg_count[seqList[row][col]] += 1

    return mw_count, bg_count


def easy_update(seqList, old_r, new_r, W, row_number, mw_count,
                bg_count):  # update counts after moving one cell to the right
    # print("easy_update")
    # print("row_number",row_number)
    # print("old_col", old_r)
    # print("new_col", new_r)
    # print("before")
    # print("bgcount", bg_count)
    # print("mwcount", mw_count)
    bg_count[seqList[row_number][old_r]] += 1
    bg_count[seqList[row_number][new_r + W - 1]] -= 1
    for index in range(W):
        mw_count[index][seqList[row_number][old_r + index]] -= 1
        mw_count[index][seqList[row_number][new_r + index]] += 1
    # print("after")
    # print("bgcount", bg_count)
    # print("mwcount", mw_count)
    return mw_count, bg_count


def hard_update(seqList, old_r, new_r, W, row_number, mw_count, bg_count):  # update counts after moving one row down
    # print("hard_update")
    # print("row_number", row_number)
    # print("old_col", old_r)
    # print("new_col", new_r)
    # print("before")
    # print("bgcount", bg_count)
    # print("mwcount", mw_count)
    if old_r == new_r:
        return mw_count, bg_count
    for index in range(W):
        mw_count[index][seqList[row_number][old_r + index]] -= 1
        mw_count[index][seqList[row_number][new_r + index]] += 1

    for index in range(W):
        bg_count[seqList[row_number][old_r + index]] += 1

    for index in range(W):
        bg_count[seqList[row_number][new_r + index]] -= 1

    # print("after")
    # print("bgcount", bg_count)
    # print("mwcount", mw_count)
    return mw_count, bg_count


def Gibbs(alphaListBg, alphaListMw, seqList, nb_iterations, W, burn_in, step):
    N = len(seqList)
    M = len(seqList[0])
    variabList = np.unique(seqList)
    # INITIALISATION
    samples = {i: 0 for i in range(M - W + 1)}
    positions_list = []
    positions_list.append(np.random.randint(0, M - W + 1, size=N))
    for it in tqdm(range(nb_iterations)):
        R = positions_list[-1].copy()
        mw_count, bg_count = init_count(seqList, R, W, variabList)
        for n in tqdm(range(N)):
            list_proba = []

            for index in range(M - W + 1):
                if index == 0:
                    mw_count, bg_count = hard_update(seqList, R[n], index, W, n, mw_count, bg_count)
                else:
                    mw_count, bg_count = easy_update(seqList, index - 1, index, W, n, mw_count, bg_count)
                R[n] = index
                proba = probaDb(alphaListBg, W, bg_count, variabList, N, M, mw_count, seqList, R, n, index) + probaDj(
                    alphaListMw, W, mw_count, variabList, N)
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
            mw_count, bg_count = hard_update(seqList, M - W, r_max, W, n, mw_count, bg_count)
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


def mean_square_error(y_true, y_pred_list):
    output = []
    for i in range(len(y_pred_list)):
        output.append(((y_pred_list[:i] - y_true) ** 2).mean())
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
    sj_square = 1. / (n - 1) * np.sum(
        np.array([(positions_list[trial, :, :] - seq_mean[trial, :]) ** 2 for trial in range(m)]), axis=1)
    W = 1. / m * np.sum(sj_square)
    Var_theta = (1 - 1. / n) * W + 1. / n * B

    R = np.sqrt(Var_theta / W)

    return R


def acc_mse(predicted, start):
    acc = 0
    for i in range(len(predicted)):
        if predicted[i] == start[i]:
            acc += 1
    acc = acc * 10
    mse = np.square(start - predicted).mean()
    return acc, mse


def choose_final(positions_list, type, start_list=None):
    trials, iterations, dimension = positions_list.shape
    if type == "last_avg":
        output = []
        for pos in range(dimension):
            position = 0
            for j in range(trials):
                position += positions_list[j, -1, pos]
            position = int(round(position / trials))
            output.append(position)
    elif type == "last_count":
        output = []
        count = {}
        for pos in range(dimension):
            count[pos] = {}
            for trial in range(trials):
                if count[pos].get(positions_list[trial, -1, pos]):
                    count[pos][positions_list[trial, -1, pos]] += 1
                else:
                    count[pos][positions_list[trial, -1, pos]] = 1
            position = max(count[pos].items(), key=lambda x: x[1])[0]
            output.append(position)
    elif type == "avg":
        output = []
        for pos in range(dimension):
            position = 0
            for trial in range(trials):
                for it in range(iterations):
                    position += positions_list[trial, it, pos]
            position = int(round(position / (trials * iterations)))
            output.append(position)
    elif type == "count":
        output = []
        count = {}
        for pos in range(dimension):
            count[pos] = {}
            for trial in range(trials):
                for it in range(iterations):
                    if count[pos].get(positions_list[trial, it, pos]):
                        count[pos][positions_list[trial, it, pos]] += 1
                    else:
                        count[pos][positions_list[trial, it, pos]] = 1

            position = max(count[pos].items(), key=lambda x: x[1])[0]
            output.append(position)
    ac_mse = (None, None) if start_list is None else acc_mse(output, start_list)
    return output, ac_mse


def majority_vote(predictions, start_list):
    options, dimension = predictions.shape
    output = []
    count = {}
    for pos in range(dimension):
        count[pos] = {}
        for trial in range(options):
            if count[pos].get(predictions[trial, pos]):
                count[pos][predictions[trial, pos]] += 1
            else:
                count[pos][predictions[trial, pos]] = 1

        position = max(count[pos].items(), key=lambda x: x[1])[0]
        output.append(position)

    return output, acc_mse(output, start_list)


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

    seq_list, start_list = generateSequences(alphaListBg, alphaListMw, num_seq, len_seq, len_motif, saveOpt=False,
                                             displayOpt=False)[:2]
    positions_list = []
    for i in range(number_trials):
        positions_list.append(Gibbs(alphaListBg, alphaListMw, seq_list, iterations, len_motif, burn_in, step))
    positions_list = np.array(positions_list)
    print("True R ", start_list)
    last_avg_pred, last_avg_acc = choose_final(positions_list, "last_avg", None)
    print("FINAL R with last_avg ", last_avg_pred, last_avg_acc)
    last_count_pred, last_count_acc = choose_final(positions_list, "last_count", start_list)
    print("FINAL R with last_count ", last_count_pred, last_count_acc)
    avg_pred, avg_acc = choose_final(positions_list, "avg", start_list)
    print("FINAL R with avg ", avg_pred, avg_acc)
    count_pred, count_acc = choose_final(positions_list, "count", start_list)
    print("FINAL R with count ", count_pred, count_acc)
    predictions = []
    predictions.append(last_avg_pred)
    predictions.append(last_count_pred)
    predictions.append(avg_pred)
    predictions.append(count_pred)
    predictions = np.array(predictions)
    majority_pred, majority_acc = majority_vote(predictions, start_list)
    print("FINAL R with majority ", majority_pred, majority_acc)

    #### Predicted Positions #####
    for m in range(num_seq):
        fig = plt.figure(figsize=(16, 8))
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
        fig = plt.figure(figsize=(16, 8))
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
    plt.figure(figsize=(16, 8))
    for m in range(num_seq):
        plt.plot(conv[:, m], label='$position = {}$'.format(m))
    plt.title('Estimated potential scale reduction over sample number')
    plt.legend(loc=1)
    plt.savefig('Figures/estimated_potential_scale_reduction.png')

    # plt.show()


def task_22():
    with open('StatMeth_Assignment2/Task_21_22/data/2_1_alphaBg.txt', 'r') as alphaBg_file:
        print("reading alpha Backgoround")
        alphaListBg = []
        for element in alphaBg_file:
            alphaListBg.append(float(element[:-1]))

    with open('StatMeth_Assignment2/Task_21_22/data/2_1_alphaMw.txt', 'r') as alphaMw_file:
        print("reading alpha Magic")
        alphaListMw = []
        for element in alphaMw_file:
            alphaListMw.append(float(element[:-1]))

    with open('StatMeth_Assignment2/Task_21_22/data/2_1_data.txt', 'r') as data_file:
        print("reading data")
        seq_list = []
        num_seq = 0
        for line in data_file:
            seq_list.append([ch for ch in (line[:-1].split())])
            num_seq += 1
            if num_seq == 24:
                break
        seq_list = np.array(seq_list)

    len_motif = 10  # needs to be read from user/file
    iterations = 100
    burn_in = 20
    step = 5
    number_trials = 10
    # start_list = [] # needs to be read from user/file
    # seq_list, start_list = generateSequences(alphaListBg, alphaListMw, num_seq, len_seq, len_motif, saveOpt=False, displayOpt=False)[:2]
    # print("real r is ",start_list)
    positions_list = []
    for i in tqdm(range(number_trials)):
        print("gibbs number", i)
        positions_list.append(Gibbs(alphaListBg, alphaListMw, seq_list, iterations, len_motif, burn_in, step))

    positions_list = np.array(positions_list)

    last_avg_pred, last_avg_acc = choose_final(positions_list, "last_avg")
    print("FINAL R with last_avg ", last_avg_pred, last_avg_acc)
    last_count_pred, last_count_acc = choose_final(positions_list, "last_count")
    print("FINAL R with last_count ", last_count_pred, last_count_acc)
    avg_pred, avg_acc = choose_final(positions_list, "avg")
    print("FINAL R with avg ", avg_pred, avg_acc)
    count_pred, count_acc = choose_final(positions_list, "count")
    print("FINAL R with count ", count_pred, count_acc)
    predictions = []
    predictions.append(last_avg_pred)
    predictions.append(last_count_pred)
    predictions.append(avg_pred)
    predictions.append(count_pred)
    predictions = np.array(predictions)
    majority_pred, majority_acc = majority_vote(predictions)
    print("FINAL R with majority ", majority_pred, majority_acc)

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
        plt.savefig('q2predicted_pos{}.png'.format(m))

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
    plt.savefig('q2estimated_potential_scale_reduction.png')

    plt.show()


if __name__ == "__main__": task_22()
