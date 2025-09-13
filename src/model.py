"""
Machine Learning model for resource allocation prediction in smart governance.

This module implements training, evaluation, and prediction functionality
for resource allocation based on regional characteristics.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODEL_CONFIG, MODEL_PATH, MODELS_DIR
from etl import DataETL, get_train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceAllocationModel:
    """
    Machine Learning model for predicting optimal resource allocation
    based on regional demographics and socioeconomic indicators.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model to use ('random_forest' or 'linear_regression')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}
        self.etl = DataETL()
        
        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=MODEL_CONFIG["n_estimators"],
                max_depth=MODEL_CONFIG["max_depth"],
                random_state=MODEL_CONFIG["random_state"]
            )
        elif model_type == "linear_regression":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "train_r2": r2_score(y_train, y_train_pred)
        }
        
        # Store feature importance for random forest
        if self.model_type == "random_forest":
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.training_metrics = metrics
        logger.info(f"Training completed. R² score: {metrics['train_r2']:.4f}")
        
        return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "test_mse": mean_squared_error(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred)
        }
        
        logger.info(f"Evaluation completed. R² score: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_single(self, region: str, district: str, population: int,
                      income_per_capita: float, health_index: float,
                      education_index: float) -> float:
        """
        Make a single prediction for a given set of inputs.
        
        Args:
            region: Region name
            district: District name
            population: Population count
            income_per_capita: Income per capita
            health_index: Health index (0-1)
            education_index: Education index (0-1)
            
        Returns:
            Predicted resource allocation
        """
        # Create a DataFrame with the input
        input_data = pd.DataFrame({
            'region': [region],
            'district': [district],
            'population': [population],
            'income_per_capita': [income_per_capita],
            'health_index': [health_index],
            'education_index': [education_index],
            'current_allocation': [0],  # Dummy value
            'year': [2023]  # Current year
        })
        
        # Process through ETL pipeline
        processed_data = self.etl.transform(input_data)
        X, _ = self.etl.prepare_features(processed_data)
        
        # Make prediction
        prediction = self.predict(X)[0]
        
        return prediction
    
    def save_model(self, model_path: Optional[str] = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model. If None, uses default path.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        if model_path is None:
            model_path = MODEL_PATH
        
        # Ensure models directory exists
        MODELS_DIR.mkdir(exist_ok=True)
        
        # Save model and ETL objects
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'etl': self.etl
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: Optional[str] = None) -> 'ResourceAllocationModel':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model. If None, uses default path.
            
        Returns:
            Loaded model instance
        """
        if model_path is None:
            model_path = MODEL_PATH
        
        try:
            model_data = joblib.load(model_path)
            
            # Create new instance
            instance = cls(model_data['model_type'])
            instance.model = model_data['model']
            instance.feature_importance = model_data['feature_importance']
            instance.training_metrics = model_data['training_metrics']
            instance.etl = model_data['etl']
            
            logger.info(f"Model loaded from {model_path}")
            return instance
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def plot_feature_importance(self, top_n: int = 10) -> None:
        """
        Plot feature importance (for random forest models).
        
        Args:
            top_n: Number of top features to display
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available for this model type")
            return
        
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            X_test: Test features
            y_test: Test target
        """
        if self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        y_pred = self.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        
        # Add R² score to plot
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


def train_and_evaluate_model(data_path: Optional[str] = None) -> ResourceAllocationModel:
    """
    Complete training and evaluation pipeline.
    
    Args:
        data_path: Path to training data
        
    Returns:
        Trained model
    """
    logger.info("Starting model training and evaluation pipeline...")
    
    # Initialize ETL and model
    etl = DataETL()
    model = ResourceAllocationModel(MODEL_CONFIG["model_type"])
    
    # Run ETL pipeline
    X, y = etl.run_pipeline(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = get_train_test_split(
        X, y, 
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    # Train model
    train_metrics = model.train(X_train, y_train)
    logger.info(f"Training metrics: {train_metrics}")
    
    # Evaluate model
    test_metrics = model.evaluate(X_test, y_test)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save model
    model.save_model()
    
    logger.info("Model training and evaluation completed successfully")
    
    return model


if __name__ == "__main__":
    # Example usage
    model = train_and_evaluate_model()
    
    # Example prediction
    prediction = model.predict_single(
        region="East",
        district="Kampala",
        population=1500000,
        income_per_capita=850,
        health_index=0.75,
        education_index=0.82
    )
    
    print(f"Predicted resource allocation: {prediction:,.0f}")