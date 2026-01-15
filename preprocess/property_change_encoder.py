import numpy as np
import pandas as pd


STEP = 0.2

def encode_property_change(input_data_path, PROPERTIES, LOG=None):
    property_change_encoder = {}
    for property_name in PROPERTIES:
        if property_name == 'pCMC':
            intervals, start_map_interval = build_intervals(input_data_path, step=STEP, LOG=LOG)
        elif property_name in ['Area_min', 'AW_ST_CMC']:
            intervals = ['low->high', 'high->low', 'no_change']

        if property_name == 'pCMC':
            property_change_encoder[property_name] = intervals, start_map_interval
        elif property_name in ['Area_min', 'AW_ST_CMC']:
            property_change_encoder[property_name] = intervals

    return property_change_encoder


def value_in_interval(value, start_map_interval):
    start_vals = sorted(list(start_map_interval.keys()))
    return start_map_interval[start_vals[np.searchsorted(start_vals, value, side='right') - 1]]


def interval_to_onehot(interval, encoder):
    return encoder.transform([interval]).toarray()[0]


def build_intervals(input_transformations_path, step=STEP, LOG=None):
    df = pd.read_csv(input_transformations_path)
    if 'Delta_pCMC' not in df.columns:
        df['Delta_pCMC'] = df['Target_Mol_pCMC'] - df['Source_Mol_pCMC']
        # save the dataframe at the location
        df.to_csv(input_transformations_path)
    delta_pCMC = df['Delta_pCMC'].tolist()
    min_val, max_val = min(delta_pCMC), max(delta_pCMC)
    if LOG:
        LOG.info("pCMC min and max: {}, {}".format(min_val, max_val))

    start_map_interval = {}
    interval_str = '({}, {}]'.format(round(-step/2, 2), round(step/2, 2))
    intervals = [interval_str]
    start_map_interval[-step/2] = interval_str

    positives = step/2
    while positives < max_val:
        interval_str = '({}, {}]'.format(round(positives, 2), round(positives+step, 2))
        intervals.append(interval_str)
        start_map_interval[positives] = interval_str
        positives += step
    interval_str = '({}, inf]'.format(round(positives, 2))
    intervals.append(interval_str)
    start_map_interval[float('inf')] = interval_str

    negatives = -step/2
    while negatives > min_val:
        interval_str = '({}, {}]'.format(round(negatives-step, 2), round(negatives, 2))
        intervals.append(interval_str)
        negatives -= step
        start_map_interval[negatives] = interval_str
    interval_str = '(-inf, {}]'.format(round(negatives, 2))
    intervals.append(interval_str)
    start_map_interval[float('-inf')] = interval_str

    return intervals, start_map_interval
