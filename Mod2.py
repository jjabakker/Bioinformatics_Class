import numpy as np
import math
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
import re
from   functools import reduce

'''
If you want to skip the preprocessing, to save time, you set preprocessing to False, 
but then you need have it run at least once.

The code file should be in a directory, e.g. 'src', and there needs to be a 'data' directory next to it
'''

preprocessing = False

def print_header(title):
    print("\n\n")
    print("---------------------------------------------------------")
    print(title)
    print("---------------------------------------------------------\n")


def find_gene(genes, gene_name):
    index_count = 0
    index_array = []
    for i in range(0,len(genes)):
        if genes[i] == gene_name:
            index_array.append(i)
            index_count += 1
    return index_array, index_count


def preprocess():

    with open('../data/dmd_data.pkl', 'rb') as f:
        datadict = pickle.load(f)

    data   = datadict['data']
    genes  = datadict['genes']
    labels = datadict['labels']


    '''
    ---------------------------------------------------------------------------
     Check for duplicate names and for strangely formatted names
     Strangely formatted is anything that contains any characters or=ther than  
     alphanumeric, -, _ or .
    
    ---------------------------------------------------------------------------
    '''

    duplicate_names = []
    strange_names = []
    already_seen = []

    for i in range(len(genes)):

        if genes[i] not in already_seen:
            already_seen.append(genes[i])
            index_array, count = find_gene(genes, genes[i])
            if count > 1:
                duplicate_names.append([genes[i], count, index_array])
            if not re.match("^[a-zA-Z0-9_\-\.]*$", genes[i]):
                strange_names.append([genes[i], count, index_array])


    # Obtain the labels of each class
    CTL = np.where(labels == 0)[0]
    DMD = np.where(labels == 1)[0]
    len_dmd   = len(DMD)
    len_ctl   = len(CTL)

    print()
    print(f"There are {len(genes)} genes in the set.")
    print(f"There are {len(labels[DMD])} genes in the DMD group and {len(labels[CTL])} in the control group.\n")

    print(f"The CTL indices are {CTL}. ")
    print(f"The DMD indices are {DMD}.")
    print()

    print(f"Data for the first 5 genes\n")
    for i in range(5):
        print("------------------------------------------------------------------")
        print(f"Data for gene: {genes[i]}")
        print("------------------------------------------------------------------\n")

        print(f"The CTL values of gene {genes[i]}:")
        print(f"{data[i][CTL]}\n")
        print(f"The average {np.average(data[i][CTL])}\n")

        print(f"The DMD values of gene {genes[i]}:")
        print(f"{data[i][DMD]}\n")
        print(f"The average {np.average(data[i][DMD])}\n")
    print()

    '''
    There are 2985 genes with a duplicate name, but with different data 
    There are also 16 genes, where the name a date is.
    
    Remove the strange names, both form the genes and data table.
    First gather all the indices in a list (flatten the list) and then 
    sort the indices revere, and delete from the end backewards to not invalidate 
    the earlier indices
    '''

    index_to_remove = []
    for strange in strange_names:
        index_to_remove.append(strange[2])

    s = reduce(lambda x,y: x+y, index_to_remove)

    for i in sorted(s, reverse = True):
        genes = np.delete(genes, i)
        data = np.delete(data, i, 0)

    # And write it
    new_data_dict = {"data": data, "genes": genes, "labels": labels}
    with open("../data/dmd_data_tidied.pkl", 'wb') as file:
            pickle.dump(new_data_dict, file, pickle.HIGHEST_PROTOCOL)

    return genes, data, labels, CTL, DMD

def read_processed():
    with open('../data/dmd_data_tidied.pkl', 'rb') as f:
        datadict = pickle.load(f)

    data = datadict['data']
    genes = datadict['genes']
    labels = datadict['labels']

    CTL = np.where(labels == 0)[0]
    DMD = np.where(labels == 1)[0]

    return genes, data, labels, CTL, DMD



'''
Now ready to start: list contains duplicates, but no weirdly names genes
'''


'''
---------------------------------------------------------------------------
 Run t-test with unequal variance
---------------------------------------------------------------------------
'''
def ttest_unequal():

    print_header("t-test with unequal variance")
    plot_results = np.zeros((nr_genes, 2), dtype=float)
    interesting_gene = []

    for i in range(nr_genes):
        t_value, p_value = st.ttest_ind(data[i][CTL], data[i][DMD], equal_var = False)
        average_CTL = np.average(data[i][CTL])
        average_DMD = np.average(data[i][DMD])
        plot_results[i][0] = math.log2(average_DMD/average_CTL)
        plot_results[i][1] = -math.log2(p_value)
        if abs(plot_results[i][0]) > 0.3 and plot_results[i][1] > 30:
            interesting_gene.append( [i, genes[i], plot_results[i][0], plot_results[i][1]])

    for ig in interesting_gene:
        print(ig[1])
        plt.text(ig[2], ig[3], ig[1])

    colors = np.random.rand(nr_genes)
    plt.title("t-test with unequal variance")
    plt.scatter(plot_results[:,0], plot_results[:,1], c = colors, alpha = 0.5)
    plt.show()


def ttest_unequal_2():

    print_header("t-test with unequal variance")
    plot_results = np.zeros((nr_genes, 2), dtype=float)
    interesting_gene = []

    [t_value, p_value] = st.ttest_ind(data[:, DMD], data[:, CTL], axis=1, equal_var = False)

    [average_CTL] = np.average(data[CTL])
    [average_DMD] = np.average(data[DMD])

    plot_results[:,0] = math.log2(average_DMD[:] / average_CTL[:])
    plot_results[:,1] = -math.log2([p_value])
    '''
    for i in range(nr_genes):

        if abs(plot_results[i][0]) > 0.3 and plot_results[i][1] > 30:
            interesting_gene.append( [i, genes[i], plot_results[i][0], plot_results[i][1]])

    for ig in interesting_gene:
        print(ig[1])
        plt.text(ig[2], ig[3], ig[1])
    
    colors = np.random.rand(nr_genes)
    plt.title("t-test with unequal variance")
    plt.scatter(plot_results[:,0], plot_results[:,1], c = colors, alpha = 0.5)
    plt.show()
'''

'''
---------------------------------------------------------------------------
 Run t-test with equal variance
---------------------------------------------------------------------------
'''

def ttest_equal():
    print_header("t-test with equal variance")
    plot_results = np.zeros((nr_genes, 2), dtype=float)
    interesting_gene = []


    for i in range(nr_genes):
        t_value, p_value = st.ttest_ind(data[i][CTL], data[i][DMD], equal_var = True)
        average_CTL = np.average(data[i][CTL])
        average_DMD = np.average(data[i][DMD])
        plot_results[i][0] = math.log2(average_DMD/average_CTL)
        plot_results[i][1] = -math.log2(p_value)
        if abs(plot_results[i][0]) > 0.3 and plot_results[i][1] > 30:
            interesting_gene.append( [i, genes[i], plot_results[i][0], plot_results[i][1]])

    for ig in interesting_gene:
        print(ig[1])
        plt.text(ig[2], ig[3], ig[1])

    colors = np.random.rand(nr_genes)
    plt.title("t-test with equal variance")
    plt.scatter(plot_results[:,0], plot_results[:,1], c = colors, alpha = 0.5)
    plt.show()

'''
---------------------------------------------------------------------------
 Run t-test with mannwhitneyu
---------------------------------------------------------------------------
'''
def mannwhitney():
    print_header("mannwhitneyu test")
    plot_results = np.zeros((nr_genes, 2), dtype=float)
    interesting_gene = []


    for i in range(nr_genes):
        u_stat, p_value = st.mannwhitneyu(data[i][CTL], data[i][DMD])
        average_CTL = np.average(data[i][CTL])
        average_DMD = np.average(data[i][DMD])
        plot_results[i][0] = math.log2(average_DMD/average_CTL)
        plot_results[i][1] = -math.log2(p_value)
        if abs(plot_results[i][0]) > 0.35 and plot_results[i][1] > 15:
            interesting_gene.append( [i, genes[i], plot_results[i][0], plot_results[i][1]])

    for ig in interesting_gene:
        print(ig[1])
        plt.text(ig[2], ig[3], ig[1])

    colors = np.random.rand(nr_genes)

    plt.title("Mannwhitneyu Test")
    plt.scatter(plot_results[:,0], plot_results[:,1], c = colors, alpha = 0.5)
    plt.show()


'''
---------------------------------------------------------------------------
 Run big test 
---------------------------------------------------------------------------
'''

def big_test():
    print_header("big test")

    original_pvalues = np.zeros(nr_genes, dtype=float)
    adjusted_pvalues = np.zeros(nr_genes, dtype=float)
    pvalues          = np.zeros(nr_genes, dtype=float)

    nr_permutations = 2
    min_pvalues = np.zeros(nr_permutations, dtype=float)

    for i in range(nr_genes):
        t_value, original_pvalues[i] = st.ttest_ind(data[i][CTL], data[i][DMD], equal_var = False)


    for j in range(nr_permutations):
        print(f"Permutation {j}")
        # Generate a new (random) label vector
        permutedLabels = np.random.permutation(labels)

        # Find the indices of the ALL and AML samples
        randALL = np.where(permutedLabels == 0)[0]
        randAML = np.where(permutedLabels == 1)[0]

        for i in range(nr_genes):

            # Extract the random samples from the original data
            rALL = data[j, randALL]
            rAML = data[j, randAML]

            t_value, pvalues[i] = st.ttest_ind(rALL, rAML, equal_var=False)

        min_pvalues[i] = min(pvalues)

    for i in range(nr_genes):
        count = 0
        for j in range(nr_permutations):
            if original_pvalues[i] < min_pvalues[j]:
                count += 1
        adjusted_pvalues[i] = (1 + count) / (1 + nr_permutations)

    for i in range(nr_genes):
        if  adjusted_pvalues[i] < 0.05:
            print(f"Gene {genes[i]} at index {i} had adjusted p-value {adjusted_pvalues[i]}")
'''
Calculate for all genes the p-value of a t-test
Store those results in original-pvalues 

for i = 0 to Nr-permutations
    Permutate the geneset
    Calculate for the permutated set the p-values for all genes 
    Determine the smallest p-value
    Store that min p-value in min-p-value[i]

for i = 0 to Nr-Genes
    Count number of instances where min-p-value < original-pvalue[i]
    Calc adjusted p-value[i] as (1 + Count) / (1 + Nr-permutations)            

All genes where adjusted p-value < 0.05 are significant
'''


if __name__ == '__main__':

    do_preprocess = False

    if do_preprocess:
        genes, data, labels, CTL, DMD = preprocess()
    else:
        genes, data, labels, CTL, DMD = read_processed()
    nr_genes = len(genes)


    #ttest_equal()
    ttest_unequal()
    #mannwhitney()


    #big_test()