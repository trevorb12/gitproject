import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
     trained_model_obj_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                'Linear Regression':LinearRegression(),
                'Decision Tree':DecisionTreeRegressor(),
                'Random Forest':RandomForestRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'Ada Boost':AdaBoostRegressor(),
                'XGBoost':XGBRegressor(),
                'KNN':KNeighborsRegressor(),
                'CatBoost':CatBoostRegressor(verbose=0),
            }
           
            model_report:dict= evaluate_model(x_train = x_train, y_train = y_train, 
                                              x_test = x_test, y_test = y_test, models = models)
            

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No good model found")
            
            logging.info(f"Best model found{best_model_name}")

            save_object(
                file_path = self.model_trainer_config.trained_model_obj_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test,predicted)
            return r2
        
        except Exception as e:
            raise CustomException(e,sys)
            