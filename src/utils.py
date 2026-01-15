import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.exception import CustomException
from src.logger import setup_logger



logger = setup_logger("Utils")

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, score='accuracy'):
    try:
        model_report = {}

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Cross Validation (Using training data)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=score)
            mean_cv = np.mean(cv_scores)

            # Test Set Prediction
            y_test_pred = model.predict(X_test)
            report = classification_report(y_test, y_test_pred, output_dict=True)
            
            # Dynamic Label Detection for Classification Report
            all_labels = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
            pos_label = all_labels[-1] # Usually the positive class is the last one (e.g., '1')

            model_report[model_name] = {
                "precision": report[pos_label]['precision'],
                "recall": report[pos_label]['recall'],
                "f1": report[pos_label]['f1-score'],
                "accuracy": report['accuracy'],
                "cv_score": mean_cv
            }
            logger.info(f"Completed evaluation for {model_name}. Test F1: {model_report[model_name]['f1']:.4f}")
        
        return model_report
    
    except Exception as e:
        raise CustomException(e, sys)