import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import setup_logger
from src.utils import save_object

import os


#sys.path.append(os.getcwd())


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # Responsible for creating the data transformation pipelines    
    def get_data_transformer_object(self):
        logger= setup_logger("DataTransformation")
        logger.info("Creating data transformation pipelines.")

        try:

            numerical_columns = ['Creatinine', 'BUN', 'GFR', 'Urine_Output', 'Age', 'Protein_in_Urine', 'Water_Intake']
            categorical_columns = ['Diabetes', 'Hypertension', 'Medication']
            
            num_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logger.info("Numerical and categorical pipelines created.")
            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")  

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logger.info("Column transformer object created.")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        logger= setup_logger("DataTransformation")
        logger.info("Entered the data transformation method or component.")

        try:
            # Reading training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data completed.")


            logger.info("Obtaining preprocessor object.")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "CKD_Status"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]


            logger.info("Applying preprocessing object on training and testing dataframes.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("Saving preprocessor object.")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)

            logger.info("Data transformation completed successfully.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logger.error("Error occurred during data transformation")
            raise CustomException(e, sys)