import typing as t
from pathlib import Path
import joblib
import pandas as pd

from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def load_dataset(*, filename:str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{filename}"))

    return dataframe

def data_preprocess(*, dataframe:pd.DataFrame):
     
    # Create features and labels 
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values

    # Create the Train Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=config.model_config.test_size,
        random_state=config.model_config.random_state
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


def save_model(*, model_file:str):

    save_file_name = f"{config.app_config.save_model_file}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    joblib.dump(model_file, save_path)    

def load_model(*, filename:str) -> None:

    file_path = TRAINED_MODEL_DIR / filename
    trained_model = joblib.load(filename=file_path)

    return trained_model    