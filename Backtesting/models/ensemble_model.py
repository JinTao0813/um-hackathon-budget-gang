import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class EnsembleModel:
    """
    Ensemble machine learning model for combining trading signals.
    """
    
    def __init__(self, model_type='random_forest', training_period=60):
        """
        Initialize the ensemble model.
        
        Parameters:
        -----------
        model_type : str
            Type of ensemble model: 'random_forest' or 'gradient_boosting'
        training_period : int
            Number of days to use for training the model
        """
        self.model_type = model_type
        self.training_period = training_period
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
    
    def _initialize_model(self):
        """Initialize the ensemble model based on the specified type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, features, target):
        """
        Train the ensemble model.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Features generated from multiple strategies
        target : pandas.Series
            Target variable (future returns direction)
            
        Returns:
        --------
        bool
            True if training was successful
        """
        if features is None or len(features) < self.training_period:
            return False
            
        # Use only training period data
        X = features.iloc[-self.training_period:].values
        y = target.iloc[-self.training_period:].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train model
        self.model = self._initialize_model()
        self.model.fit(X_scaled, y)
        self.trained = True
        return True
    
    def predict(self, features):
        """
        Generate predictions using the trained model.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            Features generated from multiple strategies
            
        Returns:
        --------
        numpy.ndarray
            Model predictions
        """
        if not self.trained or features is None:
            return None
            
        # Scale features and predict
        X_scaled = self.scaler.transform(features.values)
        return self.model.predict(X_scaled)
