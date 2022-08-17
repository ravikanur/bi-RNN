#from src.utils import save_json
import time
from src import logging
from src.constants import *
from src.components.Data_Ingestion_Component import DataIngestionPrep


STAGE = "Stage01_dataingestion_prep" ## <<< change stage name 

# init logger
def main():
    obj = DataIngestionPrep()
    obj.load_data()
    obj.shuffle_and_batch_data()
    obj.encode_traindata()
    obj.save_artifacts()


if __name__ == '__main__':
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main()
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e