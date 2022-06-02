from re import T
import typing as t
import numpy as np
import pandas as pd

from classification_model.config.core import config
from classification_model.processing.data_manager import load_model, load_dataset, data_preprocess

model_file_name = f"{config.app_config.save_model_file}.pkl"
classification_model = load_model(filename=model_file_name)

data = load_dataset(filename=config.app_config.training_data_file)

X_train, X_test, y_train, y_test = data_preprocess(dataframe=data)

def make_prediction(
    *, 
    input_data: t.Union[pd.DataFrame, dict],) -> dict:

    data = pd.DataFrame(input_data)

    prediction = classification_model.predict(X=data)

    return prediction

 


 

