##Module 1+2 - script
import numpy as np
import pickle
import math
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests

# load the Golub dataset (make sure you are in the right directory)
with open('../data/dmd_data.pkl', 'rb') as f:
    datadict = pickle.load(f)
    
data = datadict['data']
genes = datadict['genes']
labels = datadict['labels']

CONTROL = np.where(labels == 0)[0]
DISEASE = np.where(labels == 1)[0]

#1 - analysis of the data distribution
avgs_c = []
for i in range(len(genes)):
    a = data[i, :]
    c_genes = a[CONTROL]
    avg_c = np.mean(c_genes)
    avgs_c.append(avg_c)

avgs_d = []
for i in range(len(genes)):
    b = data[i, :]
    d_genes = b[DISEASE]
    avg_d = np.mean(d_genes)
    avgs_d.append(avg_d)
    
plt.subplot(1, 2, 1)
plt.hist(avgs_c)
plt.xlabel("Distribution of the control dataset")
plt.subplot(1, 2, 2)
plt.hist(avgs_d)
plt.xlabel("Distribution of the DMD dataset")
plt.show

print("The control and and patient datasets both have a Poisson distribution.")

#2- FC calculation  
FCs = []
p_values= []
for i in range(len(genes)):
    FC = math.log2(avgs_d[i]/avgs_c[i])
    FCs.append(FC)
    t_value, p_value = st.ttest_ind(data[i][DISEASE], data[i][CONTROL], equal_var= False)
    p_values.append(p_value)

#3- Multiple testing correction
#p_adjusted = multipletests(p_values, alpha=0.05 , method='fdr_bh', is_sorted=False, returnsorted=False)
p_adjusted = multipletests(p_values, alpha=0.05 , method='bonferroni') 

threshold = -math.log10(0.05) 
sign_diff_genes = []

for i, txt in enumerate(genes):
    x = FCs[i]
    y = -math.log10(p_adjusted[1][i])
    plt.scatter(x, y)
    plt.xlabel("log2FC")
    plt.ylabel("-log10(p-value)") 
    if y> threshold and abs(x)>0.35:
        plt.text(x+0.005, y+0.005, txt)
        sign_diff_genes.append(txt,)
    plt.show

#print(sign_diff_genes)
#['LUM', 'DMD', 'TYROBP', 'ACTC1', 'MYH3', 'MYBPH', 'MYH8', 'LYZ', 'ASPN', 'MYH8']

    
