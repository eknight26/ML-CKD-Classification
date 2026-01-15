import os
import sys
from dataclasses import dataclass

from src.utils import evaluate_models, save_object

from src.exception import CustomException
from src.logger import setup_logger

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logger = setup_logger("ModelTrainer")
        logger.info("Entered the model trainer method or component.")

        try:
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "LogisticRegression": LogisticRegression(),
                "SVC": SVC(),
                "RandomForestClassifier": RandomForestClassifier(),
                "MLPClassifier": MLPClassifier(),
                "XGBClassifier": xgb.XGBClassifier()
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            # Find the highest F1 score achieved
            best_f1_score = max([m['f1'] for m in model_report.values()])

            # Find all models that achieved that specific F1 score
            tied_models_names = [name for name, report in model_report.items() if report['f1'] == best_f1_score]

            if len(tied_models_names) > 1:
                logger.warning(f"Tied F1 score ({best_f1_score}) for models: {tied_models_names}. Breaking tie with Accuracy.")
                
                # Pick the one with the highest accuracy
                best_model_name = max(tied_models_names, key=lambda name: model_report[name]['accuracy'])
            else:
                best_model_name = tied_models_names[0]

            best_model_f1 = model_report[best_model_name]['f1']
            best_model_acc = model_report[best_model_name]['accuracy']
            best_model = models[best_model_name]

            logger.info(f"Best model: {best_model_name} (F1: {best_model_f1:.4f}, Acc: {best_model_acc:.4f})")

            # Threshold Check for F1 Score
            if best_model_f1 < 0.6:
                raise CustomException("No best model found with F1 score > 0.6", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            return best_model_name, best_model_f1
        
        except Exception as e:
            raise CustomException(e, sys)