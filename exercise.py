import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt

### Logic ###

def check_uniqueness(lst):
    """
    Check if a list contains only unique values.
    Returns True only if all values in the list are unique, False otherwise
    """
    if (len(lst) > 1):
        for i in range(len(lst)):
            curr_lst = lst[:i] + lst[i+1:]
            if lst[i] in curr_lst:
                return False
    return True

check_uniqueness([3])
check_uniqueness([1,3,2,5,4,6])
check_uniqueness([1,3,2,5,4,6,2])

def smallest_difference(array):
    """
    Code a function that takes an array and returns the smallest
    absolute difference between two elements of this array
    Please note that the array can be large and that the more
    computationally efficient the better
    """
    if (len(array) > 1):
        diff=abs(array[0]-array[1])
        for i in range(len(array)):
            for j in range(i+1,len(array)):
                cur_diff = abs(array[i]-array[j])
                if cur_diff < diff:
                    diff = cur_diff
        return diff
    return "please enter an array of at least 2 numbers"

smallest_difference([9])
smallest_difference([1,9,4,18,6])
smallest_difference([1,9,4,18,6,23,28,82,45,72,46])



### Finance and DataFrame manipulation ###

#reprocessing .csv data
prices=pd.read_csv("data\data.csv",sep=",")
prices['date']=prices['date'].astype(str)
prices.set_index('date',inplace=True)

def log_ret(prices):
    """
    Computes the logarithmic returns of a series of prices
    """
    return np.log(prices/prices.shift(1)).iloc[1:]


def macd(prices, window_short=12, window_long=26):
    """
    Code a function that takes a DataFrame named prices and
    returns it's MACD (Moving Average Convergence Difference) as
    a DataFrame with same shape
    Assume simple moving average rather than exponential moving average
    The expected output is in the output.csv file
    """
    pass


def sortino_ratio(prices):
    """
    Code a function that takes a DataFrame named prices and
    returns the Sortino ratio for each column
    Assume risk-free rate = 0
    On the given test set, it should yield 0.05457
    """
    val_init = prices.iloc[0]
    val_fin = prices.iloc[-1]
    return_total = val_fin/val_init - 1
    return_annual = (return_total+1)**(252/len(prices))-1
    vol_annual = log_ret(prices).std()*sqrt(252)
    return return_annual / vol_annual

sortino_ratio(prices)

def expected_shortfall(prices, level=0.95, maturity=1): #we add the maturity parameter, initiated at 1 year.
    """
    Code a function that takes a DataFrame named prices and
    returns the expected shortfall at a given level
    On the given test set, it should yield -0.03468
    """
    vol=log_ret(prices).std()  #*sqrt(252) : to be annualized a priori
    return -vol / (1-level)*norm.pdf(norm.ppf(level))*np.sqrt(maturity)

expected_shortfall(prices)

def visualize(prices, path='plot.png'):
    """
    Code a function that takes a DataFrame named prices and
    saves the plot to the given path
    """
    plot_prices = prices.plot()
    plot_prices.set_title('Visualize prices of SX5T Index')
    plot_prices.set_xlabel('dates')
    plot_prices.set_ylabel('SX5T Index')
    fig = plot_prices.figure
    fig.savefig(path, dpi=125)
    plt.close() 

visualize(prices)
