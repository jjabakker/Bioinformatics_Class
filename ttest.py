import math
import numpy
from scipy import stats



# Example from class slide

smoker = [54, 49, 54, 55, 53]
nsmoker = [53, 51, 51, 48, 41, 44]

def average(exp_data):
    return sum(exp_data) / len(exp_data)

def sigma(exp_data):

    sum  = 0
    mean = average(exp_data)
    for d in exp_data:
        sum = sum + (d - mean)**2
    sum = sum / (len(exp_data) - 1)
    return math.sqrt(sum)

mean_smoker = average(smoker)
mean_nsmoker = average(nsmoker)

sigma_smoker  = sigma(smoker)
sigma_nsmoker = sigma(nsmoker)

t1 = (len(smoker)-1) * sigma_smoker**2 + (len(nsmoker)-1) * sigma_nsmoker**2
t2 = len(smoker) + len(nsmoker) - 2
t3 = 1/len(smoker) + 1/len(nsmoker)
combined_sd = math.sqrt( (t1) * t3 / t2)

t_stat = (mean_smoker - mean_nsmoker) / combined_sd
dof = len(smoker) + len(nsmoker) - 2
p_value =  stats.t.sf(t_stat, dof)


# Example from permutation testing - page 70 from book

b = [-0.18, -0.1, -0.13, 0.3, -0.14]
a = [0.15, 0.84, 0.66, 0.52]

t2, p2 = stats.ttest_ind(a,b)
print(f"t = {t2:4.3f}")
print(f"p = {p2:4.3f}\n")

b = [-0.18, -0.1, -0.13, 0.3, 0.15]
a = [-0.14, 0.84, 0.66, 0.52]

t2, p2 = stats.ttest_ind(a,b)
print(f"t = {t2:4.3f}")
print(f"p = {p2:4.3f}\n")
