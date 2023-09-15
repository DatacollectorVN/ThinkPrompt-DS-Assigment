from configparser import ConfigParser
import joblib
import json
from datetime import datetime
import os
import sys
from loguru import logger
import pandas as pd
from src.models import *

def config(filename='modelzoo.ini', section='logistic-regression'):
    # create a parser
    parser = ConfigParser()

    # allow parse read uppercase
    parser.optionxform = str

    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            print(param[0])
            if '[' in param[1]:
                filter = param[1]
                filter = filter.lstrip('[').rstrip(']')
                filter = filter.split(',')
                filter = list(map(lambda x: float(x) if 48 <= (ord(x[0])) <= 57 else x, filter))
            else:
                filter = param[1]
            db[param[0]] = filter
            
    else:
        raise Exception("Section {0} not found in the {1} file".format(section, filename))

    return db

def save_results(estimator, model_type, duration, encode):
    os.makedirs("experiments", exist_ok = True)
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    base_dir = os.path.join("experiments", now)
    os.mkdir(base_dir)
    # print(estimator.cv_results_)
    _save_estimator(estimator.best_estimator_, model_type, base_dir)
    _save_meta_experiment(estimator, model_type, base_dir, duration, encode)

def _save_meta_experiment(estimator, model_type, base_dir, duration, encode):
    dct = {}
    dct['model_type'] = model_type
    dct['best_parameter'] = estimator.best_params_
    dct['best_score'] = estimator.best_score_
    dct['duration'] = duration
    dct['encode'] = encode
    with open(os.path.join(base_dir, "meta_experiment.json"), "w") as outfile:
        json.dump(dct, outfile)

def _save_estimator(best_estimator, model_type, base_dir):
    joblib.dump(best_estimator, os.path.join(base_dir, f"best_{model_type}.pkl"), compress = 1)
    
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def convert_time(start_time, end_time):
    '''
    Convert time (miliseconds) to minutes and seconds
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs

def custom_loggers():
    logger.level("BUG", no=38, color="<red>")
    logger.level("ANNOUNCE", no=38, color="<yellow>")

    return logger

def oversample(df, target_col):
    classes = df[target_col].value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df[target_col] == key]) 
    classes_sample = []
    for i in range(1,len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    
    return final_df