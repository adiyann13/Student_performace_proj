import os 
import sys
from src.exception import Custom_Exception
from src.logging import logging
import pandas as pd
import numpy as np
import pickle
import dill
from sklearn.metrics import r2_score


def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path ,  exist_ok=True)
        with open(file_path , 'wb') as flo:
            pickle.dump(obj , flo)
    except Exception as e:
        raise Custom_Exception(e,sys)
    

def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report ={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_r2 = r2_score(y_train,y_train_pred)
            test_model_r2 = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_r2

            return report 
        
    except Exception as e:
        raise Custom_Exception(e,sys)