from src.evaluator import Evaluator
from src.utils import custom_loggers
import os
import yaml
import gc
import warnings
from config import custom_config
warnings.filterwarnings("ignore")

FILE_ETL_CONFIG = os.path.join("config", "evaluate.yaml")
with open(FILE_ETL_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

params.update(custom_config)

# get beatiful logger
logger = custom_loggers()

def main():
    evaluator = Evaluator(logger, **params)
    if params['IS_EVAL']:
        logger.log("ANNOUNCE", "Start evaluating")
        evaluator.evaluate()
        logger.log("ANNOUNCE", "Completed evaluating")
    else:
        logger.log("ANNOUNCE", "Start infrences")
        evaluator.infer(proba = True, return_csv = True)
        logger.log("ANNOUNCE", "Completed inference") 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # release memory
        gc.collect()
        logger.log("ANNOUNCE", "Shut down server and release memory")