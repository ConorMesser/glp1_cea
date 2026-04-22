"""
Utility functions to perform CEA analyses.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def rate_to_prob(rate, time):
    """Gives probability of an event occurring over a specified time period."""
    return 1 - np.exp(-rate * time)


def prob_to_rate(prob, time):
    """Provides number of events per unit time given probability."""
    return -np.log(1 - prob) / time

def prob_time_interval(prob, intervals=12, hr=1):
    return rate_to_prob(prob_to_rate(prob, intervals) * hr, 1)


def discount(value, time, rate=0.03):
    return value / ((1 + rate) ** time)


def gen_cea(cea_input, sort_by='Cost'):
    if isinstance(cea_input, dict):
        cea_input = pd.DataFrame(cea_input).T

    assert "QALY" in cea_input.columns and "Cost" in cea_input.columns, \
        "CEA DataFrame must contain 'QALY' and 'Cost' columns."

    if 'SOC' in cea_input.index:
        cea_input = pd.concat([cea_input.loc[['SOC']], cea_input.drop('SOC').sort_values(sort_by)])
    else:
        cea_input.sort_values(sort_by, inplace=True)

    cea_input['Delta_Cost'] = cea_input['Cost'].diff().fillna(cea_input['Cost'])
    cea_input['Delta_QALY'] = cea_input['QALY'].diff().fillna(cea_input['QALY'])
    cea_input['ICER'] = cea_input['Delta_Cost'] / cea_input['Delta_QALY']

    return cea_input


def plot_cea(cea_dict=None, cea_df=None):
    cea_df = gen_cea(cea_dict) if cea_dict else gen_cea(cea_df)

    sns.lineplot(data=cea_df, x='Cost', y='QALY', marker='o')

    return cea_df
