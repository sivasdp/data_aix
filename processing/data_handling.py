import pandas as pd
import os
import joblib
from config import config


def load_dataset(file_name):
    file_path=os.path.join(config.DATAPATH,file_name)
    _data=pd.read_csv(file_path)
    return _data


def save_pipeline(pipeline_to_save):
    save_path=os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save,save_path)
    print(f'Model has been saved to {save_path} as {config.MODEL_NAME}')



def load_pipeline(pipeline_to_load):
    save_path=os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    model_loaded=joblib.load(save_path)
    print(f'Model has been Loaded')
    return model_loaded

    
    