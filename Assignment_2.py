import scipy
from scipy.special import comb
from scipy.stats import chi2
import scipy.stats as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statsmodels

def Question_1_Fisher():
    Bitter_A   = 13
    Bitter_nA  = 11
    nBitter_A  = 5
    nBitter_nA =  7

    Total_Bitter  = Bitter_A + Bitter_nA
    Total_nBitter = nBitter_A + nBitter_nA
    Total_A       = Bitter_A + nBitter_A
    Total_nA      = Bitter_nA + nBitter_nA

    print(f"Bitter_A    : {Bitter_A}")
    print(f"Bitter_nA   : {Bitter_nA}")
    print(f"nBitter_A   : {nBitter_A}")
    print(f"nBitter_nA  : {nBitter_nA}")

    print(f"Total_Bitter    : {Total_Bitter}")
    print(f"Total_nBitter   : {Total_nBitter}")
    print(f"Total_A         : {Total_A}")
    print(f"Total_nA        : {Total_nA}")

    print()
    cum = 0

    for m in range (13, 18):
        t = comb(24, m) * comb(12, 18 - m) / comb(36, 18)
        print(f"Term {m - 12} = {t}")
        cum += t

    print()
    print(cum)


def Question_2_Chi_Squared():
    '''

Observed
                A           nA
     Bitter     13          11          24
     NBitter    5           7           12
                18          18          36


Expected-0
                A                          nA
     Bitter     24/36 * 18/36 * 36         24
     NBitter                               12
                18          18             36


Expected-1
                A                          nA
     Bitter     12                         24
     NBitter                               12
                18          18             36


Expected-2
                A                          nA
     Bitter     12          12             24
     NBitter    6           6              12
                18          18             36

    '''

    chi2_sum = (13-12)**2/12 + (11-12)**2/12 + (5-6)**2/6 + (7-6)**2/6
    print(chi2.sf(0.5, 1))


def Question_3():
    '''
                       |     in set          not in set   |
    ---------------------------------------------------------------
            calcium    |        6                4        |  10
            rest       |       44              446        |  490
    ---------------------------------------------------------------
                       |       50              450        |  500


                       |     in set          not in set   |
    ---------------------------------------------------------------
            apoptosis  |       20               180       |  200
            rest       |       30               270       |  300
    ---------------------------------------------------------------
                       |       50               450       |  500
    '''

    N  = 500  # total number of genes
    m  = 50   # number of selected genes
    S1 = 10   # number of genes involved in calcium signaling
    k1 = 6    # number of genes in m and involved in calcium signaling S2 = 200 # number of genes involved in apoptosis
    k2 = 20   # number of genes in m and involved in apoptosis

    cont_array_calcium   = [[6,4],[44,446]]
    cont_array_apoptosis = [[20, 180], [30, 270]]
    oddsratio, pvalue_c = st.fisher_exact(cont_array_calcium)
    oddsratio, pvalue_a = st.fisher_exact(cont_array_apoptosis)

    print(pvalue_c)
    print(f"{pvalue_a:20.15f}")



def Question_4_Multiple_Testing():

    # Load the Golub data and define ALL and AML (indices of samples) ???
    # (1) t-test on the real data
    # perform t-test for all genes

    # Load the Golub dataset (make sure you are in the right directory)
    with open('../data/golub.pkl', 'rb') as f:
        datadict = pickle.load(f)

    data   = datadict['data']
    genes  = datadict['genes']
    labels = datadict['labels']

    # Obtain the labels of each class
    ALL = np.where(labels == 0)[0]
    AML = np.where(labels == 1)[0]
    len_all  = len(ALL)
    len_aml  = len(AML)

    [T, pTtest] = st.ttest_ind(data[:, ALL], data[:, AML], axis = 1, equal_var=True)

    # count the number of differentially expressed genes
    dex_true = np.sum(pTtest < 0.05)
    print(f"Question 4-a1")
    print(f"\tThere are {dex_true} differentially expressed genes")

    # plot the histogram of the t-statistics and p-values you calculated fig = plt.figure()
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.hist(T, bins=10, edgecolor='k')
    ax.set_xlabel('T statistic value')
    ax.set_ylabel('Frequency')

    ax = fig.add_subplot(1, 2, 2)
    ax.hist(pTtest, bins=20, edgecolor='k')
    ax.set_xlabel('p-value')
    ax.set_ylabel('Frequency')
    plt.suptitle('Original data')

    # (2) t-test on the random data
    # generate a new label vector (for now all labels are zero -> ALL)
    l = np.random.permutation(labels)

    # find the indices of the ALL and ALL samples
    randALL = np.where(l == 0)[0]
    randAML = np.where(l == 1)[0]

    # apply the t-test to the random samples
    [Trand, pRand] = st.ttest_ind(data[:, randALL], data[:, randAML], axis=1, equal_var=True)

    # count the number of differentially expressed genes
    dex_rand = np.sum(pRand < 0.05)
    print(f"Question 4-a2")
    print(f"\tThere are now {dex_rand} differentially expressed genes")

    # plot the histogram of the t-statistics and p-values you calculated
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.hist(Trand, bins=10, edgecolor='k')
    ax.set_xlabel('T statistic value')
    ax.set_ylabel('Frequency')

    ax = fig.add_subplot(1, 2, 2)
    ax.hist(pRand, bins=20, edgecolor='k')
    ax.set_xlabel('p-value')
    ax.set_ylabel('Frequency')
    plt.suptitle('Randomized data')
    plt.show()

    # Bonferroni correction

    dex_bonf = np.sum(pTtest < (0.05 / len(genes)))
    print(f"Question 4-b")
    print(f"\tWith the Bonferroni correction there are now {dex_bonf} differentially expressed genes")

    # Benjamin-Hochberg correction

    c = np.zeros([len(genes), 4])
    index_arr = np.arange(0, len(genes))

    c[:,0] = pTtest
    c[:,1] = index_arr

    c = c[c[:, 0].argsort()]
    for i in range(0, len(genes)):
        c[i,2] = 0.05 * i / len(genes)
    for i in range(len(genes) - 1, 0, -1):
        if c[i,0] < c[i,2]:
            index_found = i
            break

    print(f"There are {index_found} genes significant\n")
    for i in range(0, index_found):
        ni = int(c[i,1])
        print(genes[ni])
    print(f"\n\nQuestion 4-c")
    print(f"\tThere are now {index_found} genes significant\n")





    reject, pvals_corrected, alphacSidak, alphacBonf = statsmodels.sandbox.stats.multitest.multipletests(pTtest, alpha = 0.05,
                                                                                         method='fdr_bh ',
                                                                                           is_sorted=False,
                                                                                         returnsorted=False)

    n = 1


if __name__ == '__main__':

    Question_1_Fisher()
    Question_2_Chi_Squared()
    Question_3()
    Question_4_Multiple_Testing()