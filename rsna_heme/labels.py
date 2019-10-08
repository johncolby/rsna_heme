import os
import pandas as pd

def read_labels(base_dir):
    # Read class labels .csv
    df = pd.read_csv(os.path.join(base_dir, 'stage_1_train.csv'))
    # Parse ID column
    df[['ID', 'class']] = df['ID'].str.split('_', expand = True).iloc[:,1:3]
    df['ID'] = 'ID_' + df['ID'].astype(str)
    # Pivot
    df.drop_duplicates(inplace = True)
    df = df.pivot(index = 'ID', columns = 'class', values = 'Label')
    # Exclude subjects
    df.drop('ID_6431af929', axis = 0, inplace = True) # corrupt PixelData
    # Encode all types of hemorrhage, for stratified CV
    df.loc[:, 'cv_group'] = df.loc[:, df.columns != 'any'].astype(str).apply(lambda x: ''.join(x), axis=1)
    return df