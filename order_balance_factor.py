## --- XU YAXUAN --- ##

import pandas as pd
import numpy as np
import sys
sys.setrecursionlimit(20000)

tick_b = [f'buying_volume{i}' for i in range(1,11)]

tick_a = [f'selling_volume{i}' for i in range(1,11)]

def voi(tick_data):
    """
    Volume Order Imbalance
    Input
        - tick_data: tick_data that has been renamed
    Return
        - multiple index series (<'pandas.core.series.Series'>)
    """
    
    
    def voi_ind(tick_data_ind):
        """
        Volume Order Imbalance (Individually)
        Calculate voi at a specific tick_time for all stock_code
        """
        data = tick_data_ind
        data.index = data['tick_time']
        # delta of bid price and ask price
        sub_b_p = data['buying_price1'] - data['buying_price1'].shift(1)
        sub_a_p = data['selling_price1'] - data['selling_price1'].shift(1)

        # delta of bid volume and ask volume
        sub_b_v = data['buying_volume1'] - data['buying_volume1'].shift(1)
        sub_a_v = data['selling_volume1'] - data['selling_volume1'].shift(1)

        delta_b_v = sub_b_v
        delta_a_v = sub_a_v
        delta_b_v[sub_b_p < 0] = 0
        delta_a_v[sub_a_p > 0] = 0
        delta_b_v[sub_b_p > 0] = data['buying_volume1'][sub_b_p > 0]
        delta_a_v[sub_a_p < 0] = data['selling_volume1'][sub_a_p < 0]

        tick_fac_data = delta_b_v - delta_a_v
        
        return tick_fac_data
    
    ans = tick_data.groupby(by = 'stock_code').apply(func = voi_ind)
    return ans


def cal_weight_volume(tick_data_ind):
    """
    Calculate the weight with decreasing bid/ask order
    """
    data_dic = tick_data_ind
    data_dic.index = data_dic['tick_time']
    w = [1 - (i - 1) / 10 for i in range(1, 11)]
    w = np.array(w) / sum(w)
    
    tick_b = ['buying_volume1',
              'buying_volume2',
              'buying_volume3',
              'buying_volume4',
              'buying_volume5',
              'buying_volume6',
              'buying_volume7',
              'buying_volume8',
              'buying_volume9',
              'buying_volume10']
    
    tick_a = ['selling_volume1',
              'selling_volume2',
              'selling_volume3',
              'selling_volume4',
              'selling_volume5',
              'selling_volume6',
              'selling_volume7',
              'selling_volume8',
              'selling_volume9',
              'selling_volume10']
    
    w_b = sum([tick_data_ind[tick_b[i]]*w[i] for i in range(10)])
    w_a = sum([tick_data_ind[tick_a[i]]*w[i] for i in range(10)])
    
    return w_b, w_a

def w_voi(tick_data):
    """
    Weighted Volume Order Imbalance
    Input
        - tick_data: tick_data that has been renamed
    Return
        - multiple index series (<'pandas.core.series.Series'>)
    """
    
    def w_voi_ind(tick_data_ind):
        """
        Weighted Volume Order Imbalance (Individually)
        Calculate voi at a specific tick_time for all stock_code
        """
        data = tick_data_ind
        wb, wa = cal_weight_volume(tick_data_ind)

        data.index = data['tick_time']
        # delta of bid price and ask price
        sub_b_p = data['buying_price1'] - data['buying_price1'].shift(1)
        sub_a_p = data['selling_price1'] - data['selling_price1'].shift(1)


        sub_b_v = wb - wb.shift(1)
        sub_a_v = wa - wa.shift(1)
        delta_b_v = sub_b_v
        delta_a_v = sub_a_v

        delta_b_v[sub_b_p < 0] = 0
        delta_a_v[sub_a_p > 0] = 0
        delta_b_v[sub_b_p > 0] = wb[sub_b_p > 0]
        delta_a_v[sub_a_p < 0] = wa[sub_a_p < 0]

        tick_fac_data = delta_b_v - delta_a_v

        return tick_fac_data
    
    ans = tick_data.groupby(by = 'stock_code').apply(func = w_voi_ind)
    return ans



def ofi(tick_data):
    """
    MOFI
    Input
        - tick_data: tick_data that has been renamed
    Return
        - multiple index series (<'pandas.core.series.Series'>)
    """
    
    
    def ofi_ind(tick_data_ind):
        """
        MOFI (Individually)
        Calculate voi at a specific tick_time for all stock_code
        """
        data = tick_data_ind
        data.index = data['tick_time']
        # delta of bid price and ask price
        sub_b_p = data['buying_price1'] - data['buying_price1'].shift(1)
        sub_a_p = data['selling_price1'] - data['selling_price1'].shift(1)

        # delta of bid volume and ask volume
        sub_b_v = data['buying_volume1'] - data['buying_volume1'].shift(1)
        sub_a_v = data['selling_volume1'] - data['selling_volume1'].shift(1)

        delta_b_v = sub_b_v
        delta_a_v = sub_a_v
        delta_b_v[sub_b_p < 0] = - data['buying_volume1'][sub_b_p < 0]
        delta_a_v[sub_a_p > 0] = - data['selling_volume1'][sub_a_p > 0]
        delta_b_v[sub_b_p > 0] = data['buying_volume1'][sub_b_p > 0]
        delta_a_v[sub_a_p < 0] = data['selling_volume1'][sub_a_p < 0]

        tick_fac_data = delta_b_v - delta_a_v
        
        return tick_fac_data
    
    ans = tick_data.groupby(by = 'stock_code').apply(func = ofi_ind)
    return ans


def mofi(tick_data):
    """
    Weighted MOFI
    Input
        - tick_data: tick_data that has been renamed
    Return
        - multiple index series (<'pandas.core.series.Series'>)
    """
    
    def mofi_ind(tick_data_ind):
        """
        Weighted MOFI (Individually)
        Calculate voi at a specific tick_time for all stock_code
        """
        data = tick_data_ind
        wb, wa = cal_weight_volume(tick_data_ind)

        data.index = data['tick_time']
        # delta of bid price and ask price
        sub_b_p = data['buying_price1'] - data['buying_price1'].shift(1)
        sub_a_p = data['selling_price1'] - data['selling_price1'].shift(1)


        sub_b_v = wb - wb.shift(1)
        sub_a_v = wa - wa.shift(1)
        delta_b_v = sub_b_v
        delta_a_v = sub_a_v

        delta_b_v[sub_b_p < 0] = - wb[sub_b_p < 0]
        delta_a_v[sub_a_p > 0] = - wa[sub_a_p > 0]
        delta_b_v[sub_b_p > 0] = wb[sub_b_p > 0]
        delta_a_v[sub_a_p < 0] = wa[sub_a_p < 0]

        tick_fac_data = delta_b_v - delta_a_v

        return tick_fac_data
    
    ans = tick_data.groupby(by = 'stock_code').apply(func = mofi_ind)
    return ans

def oir(tick_data):
    """
    Order Imbalance Ratio
    Input
        - tick_data: tick_data that has been renamed
    Return
        - multiple index series (<'pandas.core.series.Series'>)
    """
        
    def oir_ind(tick_data_ind):
        """
        Order Imbalance Ratio (Individually)
        Calculate voi at a specific tick_time for all stock_code
        """
        data = tick_data_ind
        wb, wa = cal_weight_volume(tick_data_ind)

        data.index = data['tick_time']

        tick_fac_data = (wb-wa)/(wb+wa)
        
        return tick_fac_data
    
    ans = tick_data.groupby(by = 'stock_code').apply(func = oir_ind)
    return ans


def soir(tick_data):
    """
    Step Order Imbalance Ratio
    Input
        - tick_data: tick_data that has been renamed
    Return
        - multiple index series (<'pandas.core.series.Series'>)
    """
        
    def soir_ind(tick_data_ind):
        """
        Step Order Imbalance Ratio (Individually)
        Calculate voi at a specific tick_time for all stock_code
        """
        w = [1 - (i - 1) / 10 for i in range(1, 11)]
        w = np.array(w) / sum(w)
        soir_v = lambda vb, va: (vb - va) / (vb + va)
        soir_list = np.array([np.array(soir_v(tick_data_ind[tick_b[i]], tick_data_ind[tick_a[i]])) for i in range(10)])
        tick_fac_data = np.matmul(soir_list.transpose(), w)
        tick_fac_data = pd.Series(tick_fac_data)
        tick_fac_data.index = tick_data_ind['tick_time']
        return tick_fac_data
    
    ans = tick_data.groupby(by = 'stock_code').apply(func = soir_ind)
    return ans