import os
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from src.custom_dataset import CustomInferentDataset, CustomTransformer, CustomTrainDataset
import joblib
import json
import numpy as np
import pandas as pd
from src.utils import save_results, str_to_class, convert_time

class SettingConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])
    

class Evaluator(SettingConfig):
    def __init__(self, logger, **args):
        self.logger = logger
        super(Evaluator, self).__init__(**args)
    
    def _setup_data_inference(self):
        self.dataset = CustomInferentDataset(self.TEST_DATA_PATH)
        data = self.dataset.get_full_data()
        id_ = np.asarray(data[self.ID_COL])
        return id_, data
    
    def _setup_data_evaluation(self):
        self.dataset = CustomTrainDataset(self.TEST_DATA_PATH, self.OUTPUT_COL, self.ENCODE_OUTPUT, self.DROP_NA_COL, self.OVERSAMPLE)
        return self.dataset.get_full_data()

    def _load_model(self):
        return joblib.load(os.path.join(self.BEST_MODEL_BASE_PATH, self.BEST_MODEL_FILE))
    
    def _load_encode_output(self):
        with open(os.path.join(self.BEST_MODEL_BASE_PATH, self.METADATA_MODEL_FILE)) as json_file:
            dct = json.load(json_file)
        
        if dct['encode'] == '':
            return None
        else:
            return dct['encode']
    
    def infer(self, proba = False, return_csv=False):
        id_, X = self._setup_data_inference()
        self.logger.log("ANNOUNCE", f"Loaded data from {self.TEST_DATA_PATH}")
        model = self._load_model()
        self.logger.log("ANNOUNCE", f"Loaded model from {os.path.join(self.BEST_MODEL_BASE_PATH, self.BEST_MODEL_FILE)}")
        output_order = self._load_encode_output()

        if proba:
            arr_results = model.predict_proba(X)

        if return_csv:
            id_ = np.reshape(id_, (-1, 1))

            df_result = pd.DataFrame(np.hstack((id_, arr_results)), columns = [self.ID_COL] + output_order)
            df_result[self.ID_COL] = df_result[self.ID_COL].astype(int)

            df_result.to_csv(self.OUTPUT_FILE, index = False)
        
            self.logger.log("ANNOUNCE", f"Saved result to {self.OUTPUT_FILE}")

    def evaluate(self):
        X, y = self._setup_data_evaluation()
        self.logger.log("ANNOUNCE", f"Prepaed data from {self.TEST_DATA_PATH}")
        model = self._load_model()
        score = model.score(X, y)
        self.logger.log("ANNOUNCE", f"Accuracy: {score*100:.5f}%")
    