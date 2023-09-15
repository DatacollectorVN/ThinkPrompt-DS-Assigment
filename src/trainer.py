from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from src.custom_dataset import CustomTrainDataset, CustomTransformer
from src.utils import save_results, str_to_class, convert_time
import time
import sys


class SettingConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])


class Trainer(SettingConfig):
    def __init__(self, logger, **args):
        self.logger = logger
        super(Trainer, self).__init__(**args)
    
    def _setup_data_loader(self):
        self.dataset = CustomTrainDataset(self.TRAIN_DATA_PATH, self.OUTPUT_COL, self.ENCODE_OUTPUT, self.DROP_NA_COL, self.OVERSAMPLE)
        return self.dataset.get_full_data()

    def _get_encode_output_order(self):
        return list(self.dataset.encode_classes_order)
    
    def train(self):
        transformer = CustomTransformer(self.DROP_NA_COL, self.DATE_COLS, self.DROP_COLS)
        X, y = self._setup_data_loader()
        
        encode_output_order = ''
        if self.ENCODE_OUTPUT:
            encode_output_order = self._get_encode_output_order()
        
        for model_name in self.MODELS:
            self.logger.log("ANNOUNCE", f"Using {model_name} model")
            pipeline = self._create_pipeline(transformer, model_name)
            param_grid = self.PARAM_GRID.copy()
            param = str_to_class(f"{model_name}_")
            param_grid.update(param)
            estimator = GridSearchCV(
                pipeline, 
                param_grid, 
                **self.GRID_SEARCH_CV_CONFIG
            )
            start_time = time.monotonic()
            estimator.fit(X, y)
            end_time = time.monotonic()
            mins, secs = convert_time(start_time, end_time)
            duration = f"{mins}m{secs}s"
            save_results(estimator, model_name, duration, encode_output_order)
            self.logger.log("ANNOUNCE", f"Completed training {model_name} model")
    
    def _create_pipeline(self, custom_transformer, model_name):
        preprocessor = self._pipeline()
        model = str_to_class(model_name)
        if preprocessor:
            pipeline = make_pipeline(custom_transformer, preprocessor, StandardScaler(), model())
        else:
            pipeline = make_pipeline(custom_transformer, StandardScaler(), model())
    
        return pipeline
    
    def _pipeline(self):
        column_transformers = []
        preprocessor = None
        
        if len(self.MIN_MAX_SCALER_COLS) > 0:
            numeric_transformer = Pipeline(steps = [
                ("minmax_scaler", MinMaxScaler())]
            )
        column_transformers.append((numeric_transformer, self.MIN_MAX_SCALER_COLS))
        
        if len(column_transformers) != 0:
            preprocessor = make_column_transformer(
                *column_transformers,
            )
        
        return preprocessor