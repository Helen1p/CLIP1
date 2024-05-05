import pandas as pd
import numpy as np
from scipy import stats
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def srocc(xs, ys):
    """Spearman Rank Order Correlation Coefficient"""
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    srocc_result = plcc(xranks, yranks)
    return srocc_result

def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    x, y = np.float32(x), np.float32(y)
    plcc_result = stats.pearsonr(x, y)[0]
    return np.round(plcc_result,3)

if __name__=='__main__':
    # srcc和plcc放缩结果不变
    print(srocc([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4]))
    # print(srocc([1, 2, 3, 4, 5], [1, 0.9, 0.25, 0.6, 0.4]))
    # print(srocc([0.1, 0.2, 0.3, 0.4, 0.5], [10, 9, 2.5, 6, 4]))
    # print(srocc([5, 10, 15, 20, 25], [10, 9, 2.5, 6, 4]))
    print(plcc([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4]))
    # print(plcc([0.1, 0.2, 0.3, 0.4, 0.5], [10, 9, 2.5, 6, 4]))
    # print(plcc([5, 10, 15, 20, 25], [10, 9, 2.5, 6, 4]))

