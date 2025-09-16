import os
import sys

from dataclasses import dataclass

from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import Custom_exception
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class models_trainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("spliting training and testing dataset")

            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info("completed the splitting")

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear" : LinearRegression(),
                "KNN" : KNeighborsRegressor(),
                "Adaboost" : AdaBoostRegressor(),
                "Catboost" : CatBoostRegressor(verbose=False),
                "XgbClassifier" : XGBRegressor(),
            }

            logging.info("Started evaluation")

            model_report : dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            logging.info("Ended evaluation")

            #To get best model_score from dict\
            best_model_score = max(sorted(model_report.values()))

            ## to get best model_name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Custom_exception("No best model Found")
            
            logging.info("Model Training and finding the best one is done")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)

            r2_score_val = r2_score(y_test,predicted)
            return r2_score_val
        
        except Exception as e:
            raise Custom_exception(e,sys)
