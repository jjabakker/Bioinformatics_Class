# General setup, read in data, explore data .....

import numpy as np
import math
from   math import isclose
from   scipy.special import comb
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
from   LST_Functions import ztest_1samp


def print_header(title):
    print("\n\n")
    print("---------------------------------------------------------")
    print(title)
    print("---------------------------------------------------------\n")


def own_ttest_single(data, mean, sd):
    z_stat = (np.average(data) - mean) / np.sqrt(sd**2 / len(data))
    p_value = st.norm.sf(z_stat)
    return z_stat, p_value


def sigma(exp_data):
    sum  = 0
    mean = np.average(exp_data)
    for d in exp_data:
        sum = sum + (d - mean)**2
    sum = sum / (len(exp_data) - 1)
    return math.sqrt(sum)

# Load the Golub dataset (make sure you are in the right directory)
with open('../data/golub.pkl', 'rb') as f:
    datadict = pickle.load(f)

data   = datadict['data']
genes  = datadict['genes']
labels = datadict['labels']

# set the index of gene Gdf5 (it is the 2058th row, but we start from 0)
gene_idx = 2057

# get the expression data of Gdf5
a = data[gene_idx, :]

# obtain the labels of each class
ALL = np.where(labels == 0)[0]
AML = np.where(labels == 1)[0]

# Assume you do not know the index of the gene and check if there is indeed only one

index_count = 0
gene_name = "Gdf5"

for i in range(0,len(genes)):
    if genes[i].find(gene_name) != -1:
        index = i
        index_count += 1

assert (index_count == 1), "Not a unique gene"

# index now holds the index to the gene and the data is dara[index]

datagene = data[index]
allgene  = datagene[ALL]
amlgene  = datagene[AML]
lenall   = len(allgene)
lenaml   = len(amlgene)

print(f"The ALL values of gene {gene_name} are:")
print (allgene)
print("\n")

print(f"The AML values of gene {gene_name} are:")
print (amlgene)
print("\n")

# Calculate average for Eli

cs = 0
for i in range(0,len(allgene)):
    cs += allgene[i]
ave = cs / len(allgene)

'''
Exercise 1
'''

print_header("Exercise 1")
print(f"The average of the ALL values for gene {gene_name} at index {index} is: {np.average(allgene):10.7f}")
print(f"The average of the AML values for gene {gene_name} at index {index} is: {np.average(amlgene):10.7f}")
print("\n")

# To confirm the average
print (f"The average of ALL is really {sum(allgene)/len(allgene)}")
print("\n")


'''
Question 6
'''
print_header("Question 6")

z_stat_1 = (np.average(allgene) - 0) / np.sqrt(0.25**2 / lenall)
p_value_1 = st.norm.sf(z_stat_1)
print(f"The calculated ALL value of z is {z_stat_1:10.7f} and the corresponding p-value is {p_value_1:10.7f}")
if p_value_1 < 0.05:
    c = "<"
    s = ""
else:
    c = ">"
    s = "not "
print(f"Because the p_value {c} 0.05 the effect is {s}significant\n")


z_stat_1 = (np.average(amlgene) - 0) / np.sqrt(0.25**2 / lenaml)
p_value_1 = st.norm.sf(z_stat_1)
print(f"The calculated AML value of z is {z_stat_1:10.7f} and the corresponding p-value is {p_value_1:10.7f}")
if p_value_1 < 0.05:
    c = "<"
    s = ""
else:
    c = ">"
    s = "not "
print(f"Because the p_value {c} 0.05 the effect is {s}significant\n")

# Now for fun try the function that should do the same

z_stat_1a, p_value_1a =  own_ttest_single(allgene, 0, 0.25)
#assert (z_stat_1 == z_stat_1a), "Function wrong"
#assert (p_value_1 == p_value_1a), "Function wrong"

'''
Question 7
ztest_1samp(a, popmean, sigma, alternative='two-sided', axis=0):
'''

print_header("Question 7")
z_stat_2, p_value_2 = ztest_1samp(allgene, 0, 0.25, alternative = 'greater')
print(f"The retrieved valued of z is {z_stat_2:10.7f} and the corresponding p-value is {p_value_2:10.7f}\n")

if isclose(p_value_1, p_value_2):
    print(f"The two p-values are close, so, yes, same result")


'''
Question 8

One sample t-test
'''

print_header("Question 8")
index_count = 0
gene_name   = "CCND3"

for i in range(0,len(genes)):
    if genes[i].find(gene_name) != -1:
        index = i
        index_count += 1

assert (index_count == 1), "Not a unique gene"

# index now holds the index to the gene and the data is dara[index]

datagene = data[index]
allgene  = datagene[ALL]
lenall   = len(allgene)

# In this case you have to calculate the sd

sd = sigma(allgene)
t_stat_1 = (np.average(allgene) - 0) / np.sqrt(sd**2 / lenall)
p_value_1 = st.t.sf(t_stat_1, lenall - 1)
print(f"The calculated valued of t is {t_stat_1:10.7f} and the corresponding p-value is {p_value_1:10.7f}")
if p_value_1 < 0.05:
    c = "<"
    s = ""
else:
    c = ">"
    s = "not "
print(f"Because the p_value {c} 0.05 the effect is {s}significant\n")



'''
Question 9

Note that ttest_1samp calculates double sided, not single sided.
For the comparison you need to divide the p-value by 2
'''

print_header("Question 9")
t_stat_2, p_value_2 = st.ttest_1samp(allgene, 0)
p_value_2 /= 2

print(f"The retrieved valued of t is {t_stat_2:10.4f} and the corresponding p-value is {p_value_2:10.4f}\n")
print(f"The difference in t-stat = {t_stat_1 - t_stat_2:10.7f}")
print(f"The difference in p-value = {p_value_1 - p_value_2:10.7f}")


'''
Question 10

Here a two sided t-test is run
'''

print_header("Question 10")

count = 0


index_arr_0 = np.full(len(data), -1, dtype=int )
index_i     = 0
i           = 0

for d in data:
    all_d = d[ALL]
    all_m = d[AML]

    t_value, p_value  = st.ttest_ind(all_d, all_m, equal_var = False)
    if p_value < 0.05:
        count += 1
        index_arr_0[index_i] = i
        index_i += 1

    i += 1
print(count)




'''
Exercise  3

Here a two sided test is run
'''

'''
Question  11
'''
print_header("Question 11")

# Calculate the original t-statistic for CD34

index_count = 0
gene_name   = "CCND3"

for i in range(0, len(genes)):
    if genes[i].find(gene_name) != -1:
        index = i
        index_count += 1

assert (index_count == 1), "Not a unique gene"
gene_idx = index

d     = data[index]
all_d = d[ALL]
all_m = d[AML]

original_tstat, original_pval = st.ttest_ind(all_d, all_m)

# Select the number of permutations
nperm = comb(len(all_d) + len(all_m), len(all_d))

# nperm calculated this way is way too large, so take a large, but manageable value
nperm = 1000

# This is where the permuted t scores will be stored
rand_tstat = np.zeros(nperm, )

for i in range(nperm):

    # Generate a new (random) label vector
    permutedLabels = np.random.permutation(labels)

    # Find the indices of the ALL and AML samples
    randALL = np.where(permutedLabels == 0)[0]
    randAML = np.where(permutedLabels == 1)[0]

    # Extract the random samples from the original data
    rALL = data[gene_idx, randALL]
    rAML = data[gene_idx, randAML]

    # Calculate the t-statistic between rALL and rAML for iteration i
    tstat, pval = st.ttest_ind(rALL, rAML)
    rand_tstat[i] = tstat

# Plot the histogram of the t-statistics you calculated
plt.hist(rand_tstat, edgecolor = 'k')
plt.axvline(original_tstat, c = 'r', linewidth = 3)
plt.show()


#t = rand_tstat[rand_tstat > original_tstat]



index_arr_1 = np.full(len(data), -1, dtype=int )
index_i     = 0

count = 0
for i in range(0, nperm):
    if abs(rand_tstat[i]) >= abs(original_tstat):
        index_arr_1[index_i] = i
        index_i += 1
        count += 1

perm_pvalue = count / len(rand_tstat)

print(f"p-value is {perm_pvalue}")



'''
Exercise  4
'''



'''
Question  12
'''

print_header("Question 12")

p_count = 0
index_arr_2 = np.full(len(data), -1, dtype=int )
index_i = 0

for i in range(0, len(genes)):

    datagene = data[i]
    allgene = datagene[ALL]
    amlgene = datagene[AML]

    u_stat, p_value = st.mannwhitneyu(allgene, amlgene)
    if p_value < 0.05:
        index_arr_2[index_i] = i
        index_i += 1
        p_count += 1

print(f"The Number of genes differentially expressed according to rank-sum test: {p_count}")



'''
Question 13
'''

print_header("Question 13")


intersect = np.intersect1d(index_arr_0, index_arr_2)

# You have to substract 1 because the -1 is not a valid index

print(f"The number of intersects is {len(intersect) - 1}")
n = 1



'''
Addition  1
'''

print_header("Addition  1")

datagene = data[0]
allgene = datagene[ALL]
amlgene = datagene[AML]

allgene_1 = data[0][ALL]
amlgene_1 = data[0][AML]

length = len(allgene)
newall = np.empty([2, length])
newall[0,] = allgene
newall[1,]  = 1

length = len(amlgene)
newaml = np.empty([2, length])
newaml[0,] = amlgene
newaml[1,]  = 0

new = np.hstack((newall, newaml))


sortedArr = new[ :, new[0].argsort()]

len_all = len(allgene)
len_aml = len(amlgene)

R_all = 0
R_aml = 0

l = sortedArr.shape[1]
for i in range (0, l):
    if sortedArr[1][i] == 1:
        R_all += i
    else:
        R_aml += i

U_all = R_all - len_all * (len_all + 1)/2
U_aml = R_aml - len_aml * (len_aml + 1)/2

U = min(U_all, U_aml)
s = np.sqrt(len_all * len_aml * (len_all + len_aml + 1)/12)
m = len_all * len_aml / 2

z = (U - m) / s

u_stat, p_value = st.mannwhitneyu(allgene, amlgene)

n = 1


