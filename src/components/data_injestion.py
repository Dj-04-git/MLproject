import os
import sys
from src.exception import Custom_exception
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from src.components.data_transform import DataTransformation,DatatransformationConfig

@dataclass
class DataingestionConfig:
    train_data_path : str=os.path.join('artifacts',"train.csv")
    test_data_path : str=os.path.join('artifacts',"test.csv")
    raw_data_path : str=os.path.join('artifacts',"raw_data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataingestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entered into data Ingestion stage")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("read the dataset from the database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)

            logging.info("Trian test split initiated")
            train_set,test_set = train_test_split(df,random_state=42,test_size=0.2)

            train_set.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)
            
            logging.info("Ingestion of the data is completed")


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise Custom_exception(e,sys)
        
if __name__ =="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)