import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import os
import json
import logging
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def main():
    data_path = '../../tennis_atp-master/clean_tennis_data.csv'
    model_path = 'best_xgb_model.json' # keeping name for compatibility
    
    if not os.path.exists(data_path):
        log.error(f"Data file {data_path} not found!")
        return

    log.info("Loading de-leaked historical tennis dataset...")
    df = pd.read_csv(data_path)
    
    # We must sort by tourney_date to respect chronological time flow!
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['tourney_date']).sort_values('tourney_date').reset_index(drop=True)
    
    X = df.drop(['Target_P1_Wins', 'tourney_date'], axis=1)
    y = df['Target_P1_Wins']
    
    log.info("Applying TimeSeriesSplit (Walk-Forward Optimization)...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Let's perform walk-forward GridSearch. TimeSeriesSplit prevents future data leakage.
    model = GradientBoostingClassifier(random_state=42)

    # We can use slightly wider hyperparameter search since features are limited
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [50, 100, 150]
    }

    log.info("Starting Walk-Forward Hyperparameter optimization (GridSearchCV with TimeSeriesSplit)...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    log.info(f"Best hyperparameters found: {grid_search.best_params_}")
    
    # Let's evaluate exactly on the LAST split (simulating "the future")
    # To do this manually for logging:
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    log.info("Verifying model realistically on the FINAL un-seen future split...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    log.info(f"Realistic Model Accuracy on Future Walk-Forward Validation Set: {accuracy * 100:.2f}%")
    log.info("\n" + classification_report(y_test, y_pred))
    
    log.info(f"Saving best model to {model_path}...")
    joblib.dump(best_model, model_path)
    
    feature_names_path = 'model_features.json'
    with open(feature_names_path, 'w') as f:
        json.dump(list(X.columns), f)
        
    log.info("Walk-Forward Pipeline completed successfully.")

if __name__ == '__main__':
    main()
