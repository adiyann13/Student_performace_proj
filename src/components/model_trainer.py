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
            
            params={
                'dtree':{
                    'criterion':['squared_error','friedman_mse','absolute_error']
                },
                "Randomforest":{
                    'n_estimators':[120,140,150]
                },
                'AdaBoostRegressor':{
                    'learning_rate':[.1,.01,0.05,0.1],
                    'n_estimators':[23,45,89,100]
                },
                'linear_reg':{},
                'XGBRegressor':{
                    'learning_rate':[.1,0.1,.05,.001],
                    'n_estimators':[8,16,32,50,70,100]
                },
                'Gradient_boost':{
                    'learning_rate':[.1,0.1,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[70,90,120,140]
                }

            }


            model_report:dict = evaluate_models(x_train=x_train ,y_train=y_train,x_test=x_test,y_test=y_test ,models=mods, params=params)

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



