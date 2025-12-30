import os 
import sys
from src.exception import Custom_Exception
from src.logging import logging
import pandas as pd
import numpy as np
import pickle
import dill


def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path ,  exist_ok=True)
        with open(file_path , 'wb') as flo:
            pickle.dump(obj , flo)
    except Exception as e:
        raise Custom_Exception(e,sys)