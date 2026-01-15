import os
import sys

sys.path.append(os.getcwd())

from src.exception import CustomException 
from src.logger import setup_logger

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts", "data.csv")
    train_data_path:str=os.path.join("artifacts", "train.csv")
    test_data_path:str=os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger = setup_logger("DataIngestion")
        logger.info("Entered the data ingestion method or component.")

        try:
            # Source data path
            df = pd.read_csv("notebook/data/kidney_dataset.csv")
            logger.info("Read the dataset as dataframe.")

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info("Saved raw data to artifacts folder.")

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=33)
            logger.info("Performed train-test split.")

            # Save the train and test sets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logger.info("Saved train and test data to artifacts folder")

            logger.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    _, train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))


