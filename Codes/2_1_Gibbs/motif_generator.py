
import numpy as np

# This function generates sample sequences for the Magic Word problem.
# Inputs: numSeq (number of sequences - N), lenSeq (length of each sequence - M), lenMotif (length of motif - W)
#         alphaListBg (prior of Background dirichlet), alphaListMw (prior of Magic Word dirichlet)
def generateSequences(alphaListBg, alphaListMw, numSeq, lenSeq, lenMotif, saveOpt=True, displayOpt=False):

    # Generate thetas for Background and Motif with corresponding Dirichlet priors
    thetaBg = np.random.dirichlet(alphaListBg)
    thetaMw = np.random.dirichlet(alphaListMw, size=lenMotif) # Generates Theta_j for each motif position

    seqList = np.zeros((numSeq, lenSeq))
    startList = np.zeros(numSeq)

    for s in range(numSeq):
        # Get the starting point of motif
        r = np.random.randint(lenSeq-lenMotif+1)
        startList[s] = r

        for pos in range(lenSeq):
            # Sample from Background
            if pos < r or pos >= r+lenMotif:
                value = np.where(np.random.multinomial(1,thetaBg)==1)[0][0]
            # Sample from Motif
            else:
                j = pos - r # index of motif letter
                value = np.where(np.random.multinomial(1,thetaMw[j])==1)[0][0]

            seqList[s,pos] = value

    seqList = seqList.astype(int)
    startList = startList.astype(int)

    # Store the motifs in the sequences into a multidimensional array for debugging.
    motifList = np.zeros((numSeq,lenMotif))
    for i in range(numSeq):
        r = startList[i]
        motifList[i] = seqList[i,r:r+lenMotif]
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

# def sample_func(alphabet, categorical):
#     return alphabet[np.argmax(np.random.multinomial(1, categorical))]
#
#
# def generateSequences(alphabet, alphaListBg, alphaListMw, numSeq, lenSeq, lenMotif):
#     theta_back = np.random.dirichlet(alphaListBg)
#     # magic-word
#     thetas_magic = np.random.dirichlet(alphaListMw, lenMotif)
#     sample = []
#     positions = []
#     for i in range(numSeq):
#         seq = []
#         pos = np.random.randint(0, lenSeq - lenMotif + 1)  # random positions for magic words
#         for j in range(lenSeq):
#             if pos <= j and j < pos + lenMotif:  # sample for magic words
#                 seq.append(sample_func(alphabet, thetas_magic[j - pos]))
#             else:  # sample for background
#                 seq.append(sample_func(alphabet, theta_back))
#
#         sample.append(seq)
#         positions.append(pos)
#
#     return sample, positions


def main():
    alphaListBg = [1,1,1,1]
    alphaListMw = [.9,.9,.9,.9]
    numSeq = 5
    lenSeq = 10
    lenMotif = 5
    generateSequences(alphaListBg, alphaListMw, numSeq, lenSeq, lenMotif, saveOpt=False, displayOpt=True)
    # print("This \"motif_generator.py\" file contains the code for generating sequences who has similar motifs in them.")
    # print("For details, check the source file and Motif_Sketch.pdf file.\n")
    # print("In order to generate sequences, use the following function:")
    # print("seqList, startList, motifList, thetaBg, thetaMw = generateSequences(alphaListBg, alphaListMw, numSeq=30, lenSeq=20, lenMotif=5, saveOpt=True, displayOpt=True) \n")
    # print("You only need \"seqList\" output for your code. Other outputs might be useful for comparing your results.\n")
    # print("Also note that the values of alpha's must be positive.\n")
    
if __name__ == "__main__": main()