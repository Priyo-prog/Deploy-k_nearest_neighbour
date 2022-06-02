import numpy as np
from config.core import config

from processing.data_manager import load_dataset, save_model, data_preprocess
from sklearn.neighbors import KNeighborsClassifier

def run_training() -> None:

    # Read the training data
    data = load_dataset(filename=config.app_config.training_data_file)

    X_train, X_test, y_train, y_test = data_preprocess(dataframe=data)

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Save the model
    save_model(model_file=classifier)

if __name__ == '__main__':
    run_training()    



