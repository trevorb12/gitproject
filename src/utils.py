import os 
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
def save_object(file_path, obj):
    """ 
    Save the object to the file path
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
       

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            

            model.fit(x_train,y_train)

            
            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)
            train_model_score = r2_score(y_train,y_pred_train)
            test_model_score = r2_score(y_test,y_pred_test)
            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        return obj
    
    except Exception as e:
        return CustomException(e,sys)