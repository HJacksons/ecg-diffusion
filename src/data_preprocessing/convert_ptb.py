# %%
import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df):
    data = [wfdb.rdsamp(f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

Y = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
aggregate_df = pd.read_csv('scp_statements.csv', index_col=0)
aggregate_df = aggregate_df[aggregate_df.diagnostic == 1]
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in aggregate_df.index:
            tmp.append(aggregate_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

DIAGNOSTIC = 'NORM'

df_filtered_by_diagnostic = Y[Y['diagnostic_superclass'].apply(lambda x: DIAGNOSTIC in x)]
X = load_raw_data(df_filtered_by_diagnostic)


# %%
from pathlib import Path
def create_folder_if_not_exists(folder: str):
    if not(Path(folder).is_dir()):
        Path(folder).mkdir(parents=True, exist_ok=True)

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_split = int(X.shape[0] * train_ratio)
val_split = int(X.shape[0] * (train_ratio + val_ratio))

def create_egc_dataframe_by_int_index_in_dataframe(df_filtered_by_diagnostic, muse_analysis, X, i, offset=0):
    row = X[i - offset]
    df = pd.DataFrame(row, columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    ecg_id = df_filtered_by_diagnostic.index[i]
    for feature in ['rr', 'qrs', 'pr', 'qt', 'VentricularRate', 'R_PeakAmpl_i', 'R_PeakAmpl_ii', 'R_PeakAmpl_iii', 'R_PeakAmpl_v1', 'R_PeakAmpl_v2', 'R_PeakAmpl_v3', 'R_PeakAmpl_v4', 'R_PeakAmpl_v5', 'R_PeakAmpl_v6']:
        x = muse_analysis[muse_analysis['ptbecgid'] == ecg_id][feature].values[0]
        df[feature] = pd.Series(data=[x, *[None for _ in range(4999)]], dtype='str')
    return df

muse_analysis = pd.read_csv('MUSE_ground_truth.csv', sep=';', low_memory=False)
# %%
create_folder_if_not_exists('train')
train_X = X[:train_split]
for i in range(train_split):
    df = create_egc_dataframe_by_int_index_in_dataframe(df_filtered_by_diagnostic, muse_analysis, train_X, i)
    df.to_csv(f'train/{str(i).zfill(5)}.csv', index=False)
# %%
create_folder_if_not_exists('validation')
validation_X = X[train_split:val_split]
for i in range(train_split, val_split):
    df = create_egc_dataframe_by_int_index_in_dataframe(df_filtered_by_diagnostic, muse_analysis, validation_X, i, offset=train_split)
    df.to_csv(f'validation/{str(i).zfill(5)}.csv', index=False)
# %%
create_folder_if_not_exists('test')
test_X = X[val_split:]
for i in range(val_split, X.shape[0]):
    df = create_egc_dataframe_by_int_index_in_dataframe(df_filtered_by_diagnostic, muse_analysis, test_X, i, offset=val_split)
    df.to_csv(f'test/{str(i).zfill(5)}.csv', index=False)
# %%
