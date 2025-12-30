import os
import sys
from src.exception import Custom_Exception
from src.logging import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self,train_arr,test_arr):
        try:
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            mods = {
                    "dtree" : DecisionTreeRegressor(),
                    "Randomforest": RandomForestRegressor(),
                    'AdaBoostRegressor': AdaBoostRegressor(),
                    'linear_reg': LinearRegression(),
                    'XGBRegressor':XGBRegressor(),
                    'Gradient_boost': GradientBoostingRegressor(),
                    }
            
            model_report:dict = evaluate_models(x_train=x_train ,y_train=y_train,x_test=x_test,y_test=y_test ,models=mods)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = mods[best_model_name]

            save_object(file_path=self.model_trainer_config.trained_model_file_path , obj = best_model)

            prediction = best_model.predict(x_test)
            r2_score_pred = r2_score(y_test,prediction)
            return r2_score_pred
        
        except Exception as e:
            raise Custom_Exception(e,sys)



