import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors

def normalTest(testData, threshold):
    """
    returns true or false
        depending on whether testData has normal distribution

    note: inaccurate if ...
    """

    # 20<样本数<50; 用normaltest算法检验正态分布性
    if 20<len(testData)<50:
        print("use normaltest:")
        p_value = stats.normaltest(testData)[1]

    # normaltest 无法work with 20< len(testData)
    # 20<=样本数; 用Shapiro-Wilk算法检验正态分布性
    elif len(testData) < 50:
        print("use shapiro:")
        p_value= stats.shapiro(testData)[1]

    # 50<=; 用lillifors算法检验正态分布性
    elif len(testData) >=50:
        print("use lillifors:")
        p_value= lilliefors(testData)[1]


    if p_value<threshold:
        print("data are not normal distributed")
        return False
    else:
        print("data are normal distributed")
        return True
