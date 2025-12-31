import sys
import os 
from src.exception import Custom_Exception
import pandas as pd
import numpy as np
from src.logging import logging

from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def prediction(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprcessors = load_object(file_path = preprocessor_path)
            scaled_data = preprcessors.transform(features)
            preds = model.predict(scaled_data)
            return preds
        except Exception as e:
            raise Custom_Exception(e,sys)

class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender = gender 
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch =lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_df(self):
        try:
            custom_data_input = {
                "gender":[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }

            return pd.DataFrame(custom_data_input)
        
        except Exception as e:
            raise Custom_Exception(e,sys)

                   
       
    


                   


     