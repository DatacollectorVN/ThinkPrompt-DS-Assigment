from src.trainer import Trainer
from src.utils import custom_loggers
import os
import yaml
import gc
import warnings
from config import custom_config
warnings.filterwarnings("ignore")

FILE_ETL_CONFIG = os.path.join("config", "train.yaml")
with open(FILE_ETL_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

params.update(custom_config)

# get beatiful logger
logger = custom_loggers()

def main():
    trainer = Trainer(logger, **params)
    logger.log("ANNOUNCE", "Start training")
    trainer.train()
    logger.log("ANNOUNCE", "Completed training loop")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        
        # release memory
        gc.collect()
        logger.log("ANNOUNCE", "Shut down server and release memory")