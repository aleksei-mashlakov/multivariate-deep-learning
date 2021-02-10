# import tensorflow.keras.backend as K
import numpy as np
def rho_risk2(y_true, y_pred, rho):
    #
    # This function calculates the quantile risk between the true and the predicted outputs
    #
    num1 = (y_true-y_pred)*rho*(y_true>=y_pred)
    num2 = (y_pred-y_true)*(1-rho)*(y_true<y_pred)
    den  = np.sum(np.abs(y_true), axis=None)

    return 2*(np.sum(num1, axis=None)+np.sum(num2, axis=None))/den

def nrmse(y_true, y_pred):
    #
    # The formula is:
    #           K.sqrt(K.mean(K.square(y_pred - y_true)))    
    #    NRMSE = -----------------------------------------------
    #           K.mean(K.sum(y_true))
    #
    num = K.sqrt(K.mean(K.square(y_true - y_pred), axis=None))
    den = K.mean(K.abs(y_true), axis=None)

    return num / den

def rho_risk(y_true, y_pred, rho):
    #
    # This function calculates the quantile risk between the true and the predicted outputs
    #
    num = -np.sum(2*(y_pred-y_true)*(rho*(y_pred<=y_true)-(1-rho)*(y_pred>y_true)), axis=None)
    den  = np.sum(np.abs(y_true), axis=None)

    return num/den

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

def ND(y_pred, y_true):
    demoninator = np.sum(np.abs(y_true))
    diff = np.sum(np.abs(y_true - y_pred))
    return 1.0*diff/demoninator

def rmsle(y_pred, y_true) :
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_true))**2))

def NRMSE(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    denominator = np.mean(np.abs(y_true))
    diff = np.sqrt(np.mean(((y_pred-y_true)**2)))
    return diff/denominator

def rho_risk22(y_pred,y_true,rho):
    assert len(y_pred) == len(y_true)
    diff1 = (y_true-y_pred)*rho*(y_true>=y_pred)
    diff2 = (y_pred-y_true)*(1-rho)*(y_true<y_pred)
    denominator = np.sum(y_true)
    return 2*(np.sum(diff1)+np.sum(diff2))/denominator

import scipy.stats as st

def quantize(mean, var, q):
    z_score = st.norm.ppf(q)
    return mean + z_score*var

def coverage(target, quantile_forecast):
    return np.mean((target < quantile_forecast))

def PI_coverage(target, low_forecast, up_forecast):
    return np.mean((low_forecast <= target)*(target <= up_forecast))

def AEC(nominal_coverage, real_coverage):
    return (real_coverage - nominal_coverage)*100

def LR_UCoverage(target, quantile_forecast, P, cov):
    T = len(target.reshape(-1)) # total number
    NH = (np.array([target < quantile_forecast])*1).sum() #number of hits
    NZ = T - NH  #number of misses
    print(NH)
    print(T)
    c = ((1-P)**NZ)*((P)**(NH))
    t = ((1-cov)**(NZ))*((cov)**NH)

    Likelihood_Ratio = -2*np.log(c) + 2*np.log(t)
    print(1 - stats.chi2.cdf(Likelihood_Ratio, 1))
    return Likelihood_Ratio

from scipy import stats
def lr_bt(y_true, q_forecast_low, q_forecast_up, C):
    """Likelihood ratio framework of Christoffersen (1998)"""
    hits = (np.array([(q_forecast_low <= y_true)*(y_true <= q_forecast_up)])*1).reshape(-1)   # Hit series
    print(hits)
    tr = hits[1:] - hits[:-1]  # Sequence to find transitions

    # Transitions: nij denotes state i is followed by state j nij times
    n01, n10 = (tr == 1).sum(), (tr == -1).sum()
    n11, n00 = (hits[1:][tr == 0] == 1).sum(), (hits[1:][tr == 0] == 0).sum()

    # Times in the states
    n0, n1 = n01 + n00, n10 + n11
    n = n0 + n1

    # Probabilities of the transitions from one state to another
    p01, p11 = n01 / (n00 + n01), n11 / (n11 + n10)
    p = n1 / n
    print(f'p: {p}')
    print(f'n1: {n1}')
    if n1 > 0:
        # Unconditional Coverage
        uc_h0 = n0 * np.log(1 - C) + n1 * np.log(C)
        uc_h1 = n0 * np.log(1 - p) + n1 * np.log(p)
        uc = -2 * (uc_h0 - uc_h1)

        # Independence
        ind_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
        ind_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11)
        if p11 > 0:
            ind_h1 += n11 * np.log(p11)
        ind = -2 * (ind_h0 - ind_h1)

        # Conditional coverage
        cc = uc + ind

        # Stack results
        df = pd.concat([pd.Series([uc, ind, cc]),
                        pd.Series([1 - stats.chi2.cdf(uc, 1),
                                   1 - stats.chi2.cdf(ind, 1),
                                   1 - stats.chi2.cdf(cc, 2)])], axis=1)
    else:
        df = pd.DataFrame(np.zeros((3, 2))).replace(0, np.nan)

    # Assign names
    df.columns = ["Statistic", "p-value"]
    df.index = ["Unconditional", "Independence", "Conditional"]

    return df
