import os
import sys
import pandas as pd
from src.exception import Custom_Exception
from src.logging import logging
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            numric_cols = ['reading_score','writing_score']
            categ_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                            'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps =[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                    ]
            )

            categ_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy = 'most_frequent')),
                    ('OHE',OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer([
                ('numeric_pipeline', num_pipeline , numric_cols),
                ('categ_pipeline',  categ_pipeline , categ_cols)
            ])
            
            return preprocessor

            
        except Exception as e:
            raise Custom_Exception(e,sys)
        
    def initiate_data_transformation(self,train_path ,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocissing_obj = self.get_data_transformation()

            target_col = 'math_score'
            numeric_cols = ['reading_score','writing_score']

            indep_train_fts_df= train_df.drop(columns = [target_col], axis=1)
            dep_train_ft = train_df[target_col]
            indep_test_fts_df = test_df.drop(columns = [target_col], axis=1)
            dep_test_ft = test_df[target_col]

            inndep_train_ppr_arr = preprocissing_obj.fit_transform(indep_train_fts_df)
            indep_test_ppr_arr = preprocissing_obj.transform(indep_test_fts_df)

            train_arr = np.c_[inndep_train_ppr_arr , np.array(dep_train_ft)]
            test_arr = np.c_[indep_test_ppr_arr , np.array(dep_test_ft)]

            save_object(file_path = self.data_transform_config.preprocessor_obj_file_path , 
                         obj = preprocissing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise Custom_Exception(e,sys)
        


