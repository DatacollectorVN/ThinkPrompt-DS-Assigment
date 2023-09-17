import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.getcwd())

from src.utils import oversample


class CustomTrainDataset(object):
    def __init__(self, csv_file, output_col, encode_output_col=False, ref_cols_drop_na=None, is_oversample=False):
        self.data = pd.read_csv(csv_file)
        self.output_col = output_col
        self.encode_output_col = encode_output_col

        # drop null value (in test file does not include this case --> drop)
        if ref_cols_drop_na:
            self.data = self.data.dropna(subset = ref_cols_drop_na)

        if is_oversample:
            self.data = oversample(self.data, self.output_col)

        self.encode_classes_order = None
        if self.encode_output_col:
            le = LabelEncoder()
            self.y = le.fit_transform(self.data[[self.output_col]])
            self.encode_classes_order = le.classes_
        else:
            self.y = self.data[output_col]
        
        self.X = self.data.drop(output_col, axis = 1)
    
    def get_data(self, val_ratio, seed):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size = val_ratio, random_state = seed)
        return X_train, X_val, y_train, y_val

    def get_full_data(self):
        return self.X, self.y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]


class CustomInferentDataset(object):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def get_full_data(self):
        return self.data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]
    


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_na_col, date_cols, drop_cols, **args):
        self.drop_na_col =  drop_na_col
        self.date_cols = date_cols
        self.drop_cols = drop_cols
        for key in args:
            setattr(self, key, args[key])
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # feature engineering
        if self.date_cols:
            for date_col in self.date_cols:
                # get month
                X[date_col+'_month'] = X[date_col].apply(lambda x: int(x.split('-')[1]) if isinstance(x, str) else None)

                # get hour
                X[date_col+'_hour'] = X[date_col].apply(lambda x: int(x.split(' ')[1].split(':')[0]) if isinstance(x, str) else None)

        X['is_cup'] = X['is_cup'].astype(int)
        
        # drop columns
        X = X.drop(self.drop_cols, axis = 1)

        return X

if __name__ == '__main__':
    dataset = CustomTrainDataset('dataset/Predictive modeling - football-match-probability-prediction/train.csv'
                            , 'target', encode_output_col = True, is_oversample = True)
    
    # X_train, X_val, y_train, y_val = dataset.get_data(0.2, 42)
    # print(y_train)
    # t =CustomTransformer(**{'DROP_NA_COL': 'asdasd'})
    # print(t.DROP_NA_COL)