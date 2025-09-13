"""
ETL (Extract, Transform, Load) pipeline for Smart Governance data processing.

This module handles data loading, cleaning, validation, and preprocessing
for the machine learning pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Optional

from config import (
    SAMPLE_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    PREPROCESSING_CONFIG,
    VALID_REGIONS,
    VALID_DISTRICTS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataETL:
    """
    ETL pipeline for governance and policy data processing.
    
    This class handles the complete data pipeline from raw data loading
    to prepared features for machine learning models.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.data = None
        self.processed_data = None
        
    def extract(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Extract data from CSV file.
        
        Args:
            data_path: Path to data file. If None, uses default sample data.
            
        Returns:
            Raw DataFrame
        """
        if data_path is None:
            data_path = SAMPLE_DATA_PATH
            
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")
            logger.info(f"Data shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        logger.info("Starting data validation...")
        
        # Check for required columns
        required_cols = (PREPROCESSING_CONFIG["categorical_columns"] + 
                        PREPROCESSING_CONFIG["numerical_columns"] + 
                        [PREPROCESSING_CONFIG["target_column"]])
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate categorical values
        invalid_regions = set(df['region']) - set(VALID_REGIONS)
        if invalid_regions:
            logger.warning(f"Invalid regions found: {invalid_regions}")
            
        invalid_districts = set(df['district']) - set(VALID_DISTRICTS)
        if invalid_districts:
            logger.warning(f"Invalid districts found: {invalid_districts}")
        
        # Check for negative values in numerical columns
        numerical_cols = PREPROCESSING_CONFIG["numerical_columns"]
        for col in numerical_cols:
            if (df[col] < 0).any():
                logger.warning(f"Negative values found in {col}")
        
        # Check for outliers using IQR method
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            if not outliers.empty:
                logger.info(f"Found {len(outliers)} outliers in {col}")
        
        logger.info("Data validation completed")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform and clean the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting data transformation...")
        
        # Create a copy to avoid modifying original data
        transformed_df = df.copy()
        
        # Handle missing values
        if transformed_df.isnull().any().any():
            logger.info("Handling missing values...")
            # Fill numerical columns with median
            for col in PREPROCESSING_CONFIG["numerical_columns"]:
                if transformed_df[col].isnull().any():
                    median_val = transformed_df[col].median()
                    transformed_df[col].fillna(median_val, inplace=True)
            
            # Fill categorical columns with mode
            for col in PREPROCESSING_CONFIG["categorical_columns"]:
                if transformed_df[col].isnull().any():
                    mode_val = transformed_df[col].mode()[0]
                    transformed_df[col].fillna(mode_val, inplace=True)
        
        # Create additional features
        transformed_df['population_density'] = transformed_df['population'] / 1000  # Simplified density
        transformed_df['socioeconomic_index'] = (
            transformed_df['health_index'] * 0.4 + 
            transformed_df['education_index'] * 0.6
        )
        transformed_df['income_category'] = pd.cut(
            transformed_df['income_per_capita'], 
            bins=[0, 400, 600, 800, float('inf')], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Encode categorical variables
        categorical_cols = PREPROCESSING_CONFIG["categorical_columns"] + ['income_category']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            transformed_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                transformed_df[col].astype(str)
            )
        
        logger.info("Data transformation completed")
        return transformed_df
    
    def load(self, df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Load processed data to file.
        
        Args:
            df: Processed DataFrame
            output_path: Output file path. If None, uses default processed data path.
        """
        if output_path is None:
            output_path = PROCESSED_DATA_PATH
            
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for machine learning.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        feature_cols = (
            PREPROCESSING_CONFIG["numerical_columns"] + 
            ['population_density', 'socioeconomic_index'] +
            [f'{col}_encoded' for col in PREPROCESSING_CONFIG["categorical_columns"]] +
            ['income_category_encoded']
        )
        
        # Select only existing columns
        existing_feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[existing_feature_cols]
        y = df[PREPROCESSING_CONFIG["target_column"]]
        
        # Scale numerical features
        numerical_feature_cols = [col for col in existing_feature_cols 
                                 if col in PREPROCESSING_CONFIG["numerical_columns"] + 
                                 ['population_density', 'socioeconomic_index']]
        
        if numerical_feature_cols:
            X_scaled = X.copy()
            X_scaled[numerical_feature_cols] = self.scaler.fit_transform(X[numerical_feature_cols])
            return X_scaled, y
        
        return X, y
    
    def run_pipeline(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run the complete ETL pipeline.
        
        Args:
            data_path: Path to input data file
            
        Returns:
            Tuple of (features, target)
        """
        logger.info("Starting ETL pipeline...")
        
        # Extract
        raw_data = self.extract(data_path)
        
        # Validate
        validated_data = self.validate_data(raw_data)
        
        # Transform
        transformed_data = self.transform(validated_data)
        
        # Load
        self.load(transformed_data)
        
        # Prepare features
        X, y = self.prepare_features(transformed_data)
        
        self.processed_data = transformed_data
        
        logger.info("ETL pipeline completed successfully")
        return X, y


def get_train_test_split(X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    # Example usage
    etl = DataETL()
    X, y = etl.run_pipeline()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")